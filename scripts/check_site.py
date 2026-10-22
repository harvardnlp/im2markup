"""Validate public research pages and their download/citation contracts."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ORIGIN = "https://im2markup.yuntiandeng.com"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.tags = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


files = json.loads((SITE / "data/files.json").read_text())["files"]
expected = {
    "im2latex_formulas.norm.lst", "im2latex_formulas.tok.lst",
    "formula_images_processed.tar.gz", "im2latex_train_filter.lst",
    "im2latex_validate_filter.lst", "im2latex_test_filter.lst",
    "im2latex_formulas.lst", "im2latex_train.lst", "im2latex_validate.lst",
    "im2latex_test.lst", "formula_images.tar.gz", "IM2LATEX-100K-HANDWRITTEN.tgz",
}
assert {f["file"] for f in files} == expected
download_page = Page(SITE / "data/index.html")
downloads = {a["href"] for tag, a in download_page.tags if tag == "a" and "download" in a}
assert expected <= downloads, "A historical download URL was removed or changed"
assert len(files) == 12 and all(f["bytes"] > 0 and len(f["sha256"]) == 64 for f in files)
assert (SITE / "data/checksums.sha256").read_text() == "".join(f'{f["sha256"]}  {f["file"]}\n' for f in files)

bib = (SITE / "citation.bib").read_text()
for name in ("Deng, Yuntian", "Kanervisto, Anssi", "Ling, Jeffrey", "Rush, Alexander M."):
    assert name in bib
assert (ROOT / "CITATION.cff").read_bytes() == (SITE / "CITATION.cff").read_bytes()
csl = json.loads((SITE / "citation.csl.json").read_text())[0]
assert csl["author"][0] == {"family": "Deng", "given": "Yuntian"}
assert len(csl["author"]) == 4 and csl["issued"]["date-parts"] == [[2017]]

for name, url in (("index.html", "/"), ("data/index.html", "/data/"), ("demo.html", "/demo.html")):
    page = Page(SITE / name)
    assert any(t == "meta" and a.get("name") == "viewport" for t, a in page.tags)
    assert any(t == "link" and a.get("rel") == "canonical" and a.get("href") == ORIGIN + url for t, a in page.tags)
    assert sum(t == "h1" for t, _ in page.tags) == 1
    for tag, attrs in page.tags:
        for key in ("href", "src"):
            link = attrs.get(key, "")
            if not link or urlparse(link).scheme or link.startswith("//"):
                continue
            resolved = urlparse(urljoin(ORIGIN + url, link))
            target = SITE / resolved.path.lstrip("/")
            if resolved.path.startswith("/data/") and target.name in expected:
                continue  # Published data archives remain on the original server.
            if resolved.path.endswith("/"):
                target /= "index.html"
            assert target.is_file(), (name, link, "missing local asset")
    for t, attrs in page.tags:
        if t == "img":
            assert attrs.get("alt"), (name, "missing image description")

for name in ("index.html", "data/index.html"):
    text = (SITE / name).read_text()
    for href, label in re.findall(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', text, re.DOTALL):
        parsed = urlparse(href)
        if parsed.hostname != "arxiv.org":
            continue
        assert parsed.scheme == "https" and not parsed.path.endswith(".pdf"), (name, href)
        if parsed.path.startswith("/pdf/"):
            assert re.sub(r"<[^>]+>", "", label).strip() == "PDF", (name, href, "PDF link must be labeled")
    assert 'href="https://arxiv.org/abs/1609.04938v1"' in text, (name, "Original-paper link must preserve v1")
    encoded = text.split('<script type="application/ld+json">')[1].split("</script>")[0]
    structured = json.loads(encoded)
    assert "Yuntian Deng" in encoded and "Anssi Kanervisto" in encoded
    if name.startswith("data/"):
        assert structured == json.loads((SITE / "data/dataset.json").read_text())
        assert structured["license"] == "https://creativecommons.org/publicdomain/zero/1.0/"
    from html import unescape
    assert unescape(text.split('<pre id="bibtex">')[1].split("</pre>")[0]) == bib

assert '<a href="https://arxiv.org/pdf/1609.04938v1">PDF</a>' in (SITE / "index.html").read_text()
assert "https://arxiv.org/abs/1609.04938v1" in (ROOT / "README.md").read_text()

ET.parse(SITE / "sitemap.xml")
assert "Sitemap: " + ORIGIN + "/sitemap.xml" in (SITE / "robots.txt").read_text()
print("Validated historical downloads, citation consistency, structured data, and page assets.")
