"""Validate public research pages and their download/citation contracts."""
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ORIGIN = "https://im2markup.yuntiandeng.com"
TITLE_2016 = "What You Get Is What You See: A Visual Markup Decompiler"
TITLE_2017 = "Image-to-Markup Generation with Coarse-to-Fine Attention"
AUTHORS_2016 = ["Yuntian Deng", "Anssi Kanervisto", "Alexander M. Rush"]
AUTHORS_2017 = ["Yuntian Deng", "Anssi Kanervisto", "Jeffrey Ling", "Alexander M. Rush"]


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.tags = []
        self.visible_text = []
        self.hidden_depth = 0
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag in ("script", "style"):
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden_depth -= 1

    def handle_data(self, value):
        if not self.hidden_depth:
            self.visible_text.append(value)


def paper_records(value):
    """Find publication records in structured data."""
    if isinstance(value, dict):
        if value.get("@type") == "ScholarlyArticle":
            yield value
        for child in value.values():
            yield from paper_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from paper_records(child)


def check_publications(value):
    papers = {p["@id"]: p for p in paper_records(value)}
    first = papers[ORIGIN + "/#paper-2016"]
    final = papers[ORIGIN + "/#paper"]
    assert first["name"] == TITLE_2016
    assert first["datePublished"] == "2016-09-16"
    assert first["url"] == "https://arxiv.org/abs/1609.04938v1"
    assert [a["name"] for a in first["author"]] == AUTHORS_2016
    assert final["name"] == TITLE_2017 and final["datePublished"] == "2017"
    assert final["url"] == "https://proceedings.mlr.press/v70/deng17a.html"
    assert final["sameAs"] == "https://arxiv.org/abs/1609.04938v2"
    assert [a["name"] for a in final["author"]] == AUTHORS_2017
    assert final["isBasedOn"]["@id"] == first["@id"]


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
csl_records = json.loads((SITE / "citation.csl.json").read_text())
csl = csl_records[0]
assert csl["author"][0] == {"family": "Deng", "given": "Yuntian"}
assert len(csl["author"]) == 4 and csl["issued"]["date-parts"] == [[2017]]
original_csl = next(c for c in csl_records if c["id"] == "deng2016visualmarkup")
assert original_csl["title"] == TITLE_2016
assert len(original_csl["author"]) == 3
assert original_csl["issued"]["date-parts"] == [[2016, 9, 16]]
original_bib = (SITE / "citation-2016.bib").read_text()
assert TITLE_2016 in original_bib and "1609.04938v1" in original_bib
assert "Jeffrey" not in original_bib

for name, url in (("index.html", "/"), ("data/index.html", "/data/"), ("demo.html", "/demo.html"), ("guide/index.html", "/guide/")):
    page = Page(SITE / name)
    assert any(t == "meta" and a.get("name") == "viewport" for t, a in page.tags)
    assert any(t == "link" and a.get("rel") == "canonical" and a.get("href") == ORIGIN + url for t, a in page.tags)
    assert sum(t == "h1" for t, _ in page.tags) == 1
    ids = [a["id"] for _, a in page.tags if "id" in a]
    assert len(ids) == len(set(ids)), (name, "duplicate element IDs")
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
    check_publications(structured)
    assert "Yuntian Deng" in encoded and "Anssi Kanervisto" in encoded
    if name.startswith("data/"):
        assert structured == json.loads((SITE / "data/dataset.json").read_text())
        assert structured["citation"] == [
            "https://proceedings.mlr.press/v70/deng17a.html",
            "https://arxiv.org/abs/1609.04938v1",
        ], "Dataset citations must link directly to the two papers"
        assert structured["license"] == "https://creativecommons.org/publicdomain/zero/1.0/"
        assert [a["name"] for a in structured["creator"]] == AUTHORS_2016
        assert {c["@id"] for c in structured["subjectOf"]} == {ORIGIN + "/#paper-2016", ORIGIN + "/#paper"}
    assert unescape(text.split('<pre id="bibtex">')[1].split("</pre>")[0]) == bib
    assert any(t == "a" and a.get("href") == "/guide/" for t, a in Page(SITE / name).tags)

for name in ("index.html", "data/index.html", "guide/index.html"):
    visible = " ".join(Page(SITE / name).visible_text)
    assert TITLE_2016 in visible and TITLE_2017 in visible, (name, "Both paper titles must be readable")
guide = (SITE / "guide/index.html").read_text()
guide_article = json.loads(re.search(r'<script type="application/ld\+json">(.*?)</script>', guide, re.S)[1])
check_publications(guide_article)
canonical_dataset = json.loads((SITE / "data/dataset.json").read_text())
for field in ("@id", "@type", "name", "url", "description", "license", "creator"):
    assert guide_article["about"][field] == canonical_dataset[field], ("guide dataset", field)

assert '<a href="https://arxiv.org/pdf/1609.04938v1">PDF</a>' in (SITE / "index.html").read_text()
assert "https://arxiv.org/abs/1609.04938v1" in (ROOT / "README.md").read_text()

sitemap = ET.parse(SITE / "sitemap.xml")
assert ORIGIN + "/guide/" in {n.text for n in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")}
assert "Sitemap: " + ORIGIN + "/sitemap.xml" in (SITE / "robots.txt").read_text()
print("Validated historical downloads, citation consistency, structured data, and page assets.")
