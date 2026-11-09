# CPU checks for sources/fetch_wiki_html.py: stub filtering and retry handling. No network.
# Run: python -m pytest tests/
import gzip
import io
import os
import sys
import urllib.error

from lmlm.constants import ROOT_DIR

sys.path.insert(0, os.path.join(ROOT_DIR, "sources"))
import fetch_wiki_html as f  # noqa: E402

STUB = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page><title>Ada Lovelace</title><ns>0</ns><id>7</id><revision><id>700</id></revision></page>
  <page><title>Lovelace</title><ns>0</ns><id>8</id><redirect title="Ada Lovelace" /><revision><id>800</id></revision></page>
  <page><title>Talk:Ada Lovelace</title><ns>1</ns><id>9</id><revision><id>900</id></revision></page>
</mediawiki>"""


def test_stub_yields_articles_only(tmp_path):
    path = tmp_path / "stub.xml.gz"
    path.write_bytes(gzip.compress(STUB.encode()))
    assert list(f.iter_stub(path)) == [(7, "Ada Lovelace", 700)]


class FakeResponse(io.BytesIO):
    headers = {"Content-Encoding": "gzip"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serve(monkeypatch, statuses):
    """urlopen answers with these statuses in order; 200 serves gzip "<html/>". Returns the call log."""
    calls = []

    def urlopen(request, timeout):
        status = statuses[len(calls)]
        calls.append(status)
        if status == 200:
            return FakeResponse(gzip.compress(b"<html/>"))
        raise urllib.error.HTTPError(request.full_url, status, "", {"Retry-After": "1"}, None)

    monkeypatch.setattr(f.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(f.time, "sleep", lambda s: None)
    return calls


def test_rate_limits_are_retried_until_served(monkeypatch):
    calls = serve(monkeypatch, [429, 503, 200])
    limiter = f.Limiter(200)
    status, html = f.fetch(700, limiter, "test")
    assert (status, gzip.decompress(html), calls) == (200, b"<html/>", [429, 503, 200])
    assert limiter.next_slot > f.time.monotonic()  # Retry-After paused the shared limiter


def test_final_statuses_are_returned(monkeypatch):
    calls = serve(monkeypatch, [404])
    assert f.fetch(700, f.Limiter(200), "test") == (404, None) and calls == [404]  # deleted page: no retry
    calls = serve(monkeypatch, [500] * f.MAX_ATTEMPTS)
    assert f.fetch(700, f.Limiter(200), "test") == (500, None) and len(calls) == f.MAX_ATTEMPTS
