"""Fetch the rendered HTML of every article revision listed in the enwiki-20260901 stub dump.

Run: WIKIMEDIA_CONTACT=<email or URL> python sources/fetch_wiki_html.py   (from the repo root)

Reads data/raw/enwiki-20260901-stub-articles.xml.gz and keeps articles (namespace 0, redirects skipped).
Fetches each revision from https://en.wikipedia.org/w/rest.php/v1/revision/<rev_id>/html, rendered by
MediaWiki's own parser (templates expanded), and stores it in data/raw/enwiki-20260901-html.sqlite:
  pages(rev_id PRIMARY KEY, page_id, title, status, html)   html: gzip bytes, NULL unless status is 200
Resumable: revisions already in the database are skipped. Statuses other than 200 are stored and not
retried on rerun (404: page deleted since the dump).

Politeness (https://wikitech.wikimedia.org/wiki/Robot_policy,
https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits):
- User-Agent names this client and a contact (WIKIMEDIA_CONTACT), as the User-Agent policy requires.
- At most 3 requests in flight and 200 requests per minute: the REST API concurrency limit and the
  rate limit for an unauthenticated bot with a compliant User-Agent.
- 429 and 503 pause every worker for Retry-After seconds, or 5 s doubling to 15 min without one.
"""
import gzip
import os
import queue
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from lmlm.constants import DATA_DIR

STUB_PATH = os.path.join(DATA_DIR, "raw", "enwiki-20260901-stub-articles.xml.gz")
DB_PATH = os.path.join(DATA_DIR, "raw", "enwiki-20260901-html.sqlite")
URL = "https://en.wikipedia.org/w/rest.php/v1/revision/{}/html"
WORKERS = 3
# wozniak: the unauthenticated limit puts ~7M articles at ~24 days. Upgrade path: an access token from
# an established account lifts the API limit to 2000/min; the robot policy then caps REST at 10/s (600/min).
REQUESTS_PER_MINUTE = 200
BACKOFF_START, BACKOFF_MAX = 5, 900
MAX_ATTEMPTS = 5  # for 5xx other than 503 and network errors; the last status is stored
COMMIT_EVERY = 100


def iter_stub(path):
    """(page_id, title, rev_id) for every non-redirect namespace-0 page in a stub dump."""
    with gzip.open(path) as f:
        context = ET.iterparse(f, events=("start", "end"))
        _, root = next(context)
        ns = root.tag[:root.tag.index("}") + 1]
        for event, elem in context:
            if event != "end" or elem.tag != ns + "page":
                continue
            if elem.findtext(ns + "ns") == "0" and elem.find(ns + "redirect") is None:
                yield int(elem.findtext(ns + "id")), elem.findtext(ns + "title"), int(elem.findtext(f"{ns}revision/{ns}id"))
            root.clear()  # free parsed pages


class Limiter:
    """Spaces request starts across all workers; pause() delays every worker."""

    def __init__(self, per_minute):
        self.interval, self.next_slot, self.lock = 60 / per_minute, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        time.sleep(slot - now)

    def pause(self, seconds):
        with self.lock:
            self.next_slot = max(self.next_slot, time.monotonic() + seconds)


def fetch(rev_id, limiter, user_agent):
    """(status, gzip html or None) for one revision, retrying rate limits and transient failures."""
    request = urllib.request.Request(URL.format(rev_id), headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"})
    backoff, attempts = BACKOFF_START, 0
    while True:
        limiter.wait()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                return 200, body if response.headers.get("Content-Encoding") == "gzip" else gzip.compress(body)
        except urllib.error.HTTPError as e:
            status, retry_after = e.code, e.headers.get("Retry-After")
        except OSError:  # connection reset, timeout, DNS
            status, retry_after = None, None
        if status in (429, 503):
            limiter.pause(int(retry_after) if retry_after and retry_after.isdigit() else backoff)
        elif status is None or status >= 500:
            attempts += 1
            if attempts == MAX_ATTEMPTS:
                return status or 0, None
            limiter.pause(backoff)
        else:
            return status, None  # 404 deleted, 403 suppressed: final
        backoff = min(backoff * 2, BACKOFF_MAX)


def main():
    contact = os.environ.get("WIKIMEDIA_CONTACT")
    if not contact:
        raise SystemExit("Set WIKIMEDIA_CONTACT to an email or URL; Wikimedia's User-Agent policy requires one.")
    user_agent = f"LMLM-corpus-fetch/0.1 ({contact}) python-urllib"

    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS pages "
               "(rev_id INTEGER PRIMARY KEY, page_id INTEGER, title TEXT, status INTEGER, html BLOB)")
    done = {row[0] for row in db.execute("SELECT rev_id FROM pages")}
    db_lock, limiter, todo = threading.Lock(), Limiter(REQUESTS_PER_MINUTE), queue.Queue(maxsize=WORKERS)  # small: Ctrl-C stops after the in-flight requests
    counts = {"fetched": 0, "failed": 0}
    start = time.monotonic()

    def worker():
        while (item := todo.get()) is not None:
            page_id, title, rev_id = item
            status, html = fetch(rev_id, limiter, user_agent)
            with db_lock:
                db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?)", (rev_id, page_id, title, status, html))
                counts["fetched" if status == 200 else "failed"] += 1
                n = counts["fetched"] + counts["failed"]
                if n % COMMIT_EVERY == 0:
                    db.commit()
                if n % 10000 == 0:
                    print(f"{n} this run ({counts['failed']} failed), {n / (time.monotonic() - start) * 60:.0f}/min")

    threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    try:
        for page_id, title, rev_id in iter_stub(STUB_PATH):
            if rev_id not in done:
                todo.put((page_id, title, rev_id))
    finally:
        for _ in threads:
            todo.put(None)
        for t in threads:
            t.join()
        with db_lock:
            db.commit()
        print(f"done: {counts['fetched']} fetched, {counts['failed']} failed this run -> {DB_PATH}")


if __name__ == "__main__":
    main()
