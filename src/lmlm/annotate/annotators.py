import json
import os
import re
from openai import OpenAI

from copy import deepcopy
from lmlm.constants import PROMPTS_DIR, CONFIGS_DIR
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from lmlm.annotate.context_budget import budget_prompts, measure_max_model_len


class Prompt:
    def __init__(self, prompt_id):
        self.prompt_id = prompt_id
        path = os.path.join(PROMPTS_DIR, f"{prompt_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found at {path}.")
        with open(path, "r") as f:
            self.prompt = json.load(f)

    def __call__(self, text):
        """
        Fills the placeholders of the form [INSERT_TEXT] in the prompt with the provided text.

        Args:
            text (str): The text to insert into the prompt.

        Returns:
            list: A list of dictionaries with the 'content' field updated to include the provided text.
        """
        filled_prompt = deepcopy(self.prompt)
        for prompt_dict in filled_prompt:
            try:
                prompt_dict['content'] = re.sub(
                    r'\[INSERT_TEXT\]',
                    text,  # Remove the lambda function - it's unnecessary here
                    prompt_dict['content']
                )
            except re.error as e:
                if 'INSERT_TEXT' in filled_prompt[-1]['content']:
                    filled_prompt[-1]['content'] = text
        return filled_prompt

class Annotator:
    def __init__(self, model_id, prompt_id, config_file):
        self.model_id = model_id
        self.prompt = Prompt(prompt_id)
        if config_file is not None:
            with open(os.path.join(CONFIGS_DIR, f"{config_file}.json"), "r") as f:
                self.configs = json.load(f)
        else:
            self.configs = {}

    def annotate(self, texts):
        raise NotImplementedError

    def postprocess(self, texts):
        raise NotImplementedError

class ChatGPTAnnotator(Annotator):
    def __init__(self, model_id, prompt_id, config_file):
        super().__init__(model_id, prompt_id, config_file)
        self.client = OpenAI()

    def annotate(self, texts):
        annotations = []
        for text in texts:
            message = self.prompt(text)

            annotations.append(
                self.client.chat.completions.create(
                model=self.model_id,
                messages=message
            ).choices[0].message.content
            )
        return annotations


class LlamaAnnotator(Annotator):
    def __init__(self, model_id, prompt_id, config_file):
        super().__init__(model_id, prompt_id, config_file)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.max_model_len = measure_max_model_len(self.model_id, self.configs['llm'])
        self.llm = LLM(model=self.model_id, max_model_len=self.max_model_len, **self.configs['llm'])
        self.dropped = []  # texts skipped by the last annotate() call

    def annotate(self, texts):
        """One annotation per text; None where the text was dropped (reason recorded in self.dropped)."""
        kept, self.dropped = budget_prompts(texts, self.tokenizer, self.prompt, self.max_model_len)
        params = [SamplingParams(**{**self.configs['sampling'], "max_tokens": max_tokens}) for _, _, max_tokens in kept]

        try:
            responses = self.llm.generate([p for _, p, _ in kept], params)
        except Exception as e:
            print(f"Error occurred: {e}")
            return []

        annotated_texts = [None] * len(texts)
        for (i, _, _), response in zip(kept, responses):
            output = response.outputs[0]
            if output.finish_reason == "length":
                # The output reached max_model_len; keep no partial annotation.
                self.dropped.append({"index": i, "tokens_needed": None, "max_model_len": self.max_model_len,
                                     "reason": "output reached max_model_len"})
            else:
                annotated_texts[i] = output.text
        return annotated_texts

    
class LlamaLoraAnnotator(Annotator):
    def __init__(self, model_id, prompt_id, config_file):
        super().__init__(model_id, prompt_id, config_file)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.llm = LLM(model=self.configs['base_model'], **self.configs['llm'])
        self.sampling_params = SamplingParams(**self.configs['sampling'])

    def annotate(self, texts=None, prompts=None):
        assert prompts or texts, "Either `texts` or `prompts` must be provided."
        if prompts is None:
            prompts = [
                self.tokenizer.apply_chat_template(
                    self.prompt(text),
                    tokenize=False,
                    add_generation_prompt=True
                ).removeprefix(self.tokenizer.bos_token)  # vLLM adds <|begin_of_text|> when it encodes the string
                for text in texts
            ]
        responses = self.llm.generate(prompts, self.sampling_params, lora_request=LoRARequest("lora_adapter", 1, self.model_id), use_tqdm=True)
        annotated_texts = list(map(lambda x: x.outputs[0].text, responses))
        return annotated_texts



if __name__ == "__main__":
    print("Testing the Prompt class.")
    prompt = Prompt("claude-v0")
    results = prompt("This is some random sample text to test the class.")
    print(results)
    print("\n\n")

    print("Testing the LlamaAnnotator class.")
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama_annotator = LlamaAnnotator(model_id, "claude-v0", "llama/default")
    texts = [
        "Friedrich Wilhelm Nietzsche (15 October 1844 – 25 August 1900) was a German classical scholar, philosopher, and critic of culture, who became one of the most influential of all modern thinkers.",
        "Theodore Roosevelt Jr. (October 27, 1858 – January 6, 1919), also known as Teddy or T. R., was the 26th president of the United States, serving from 1901 to 1909."
    ]
    annotated_texts = llama_annotator.annotate(texts)
    for text in annotated_texts:
        print(text)
