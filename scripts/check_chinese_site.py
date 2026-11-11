"""Validate the Chinese edition's language, attribution and download contracts.

Run from any directory with Python's standard library. No network is required.
"""
import datetime as dt
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ORIGIN = "https://im2markup.yuntiandeng.com"
PAIRS = (("/zh/", "/"), ("/zh/data/", "/data/"), ("/zh/guide/", "/guide/"))
TITLES = {
    ORIGIN + "/#paper-2016": "What You Get Is What You See: A Visual Markup Decompiler",
    ORIGIN + "/#paper": "Image-to-Markup Generation with Coarse-to-Fine Attention",
}
AUTHORS = {
    ORIGIN + "/#paper-2016": ["Yuntian Deng", "Anssi Kanervisto", "Alexander M. Rush"],
    ORIGIN + "/#paper": ["Yuntian Deng", "Anssi Kanervisto", "Jeffrey Ling", "Alexander M. Rush"],
}
DOWNLOADS = {
    "IM2LATEX-100K-HANDWRITTEN.tgz", "formula_images.tar.gz",
    "formula_images_processed.tar.gz", "im2latex_formulas.lst",
    "im2latex_formulas.norm.lst", "im2latex_formulas.tok.lst",
    "im2latex_test.lst", "im2latex_test_filter.lst", "im2latex_train.lst",
    "im2latex_train_filter.lst", "im2latex_validate.lst", "im2latex_validate_filter.lst",
}
# Translations share the published citations; they must not rewrite the originals.
BIBTEX_SHA256 = {
    "citation.bib": "caf6db5fad41e7ca596afe500c1cf20e424ae32fc713a8f10e0df3d3001e89f8",
    "citation-2016.bib": "95249f0aa4aa876158b0e0080b29849da7f8621d5216ed34d89a50665cbe5d9a",
}


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.tags = []
        self.visible = []
        self.jsonld = []
        self.pre_blocks = []
        self._hidden = None
        self._script_type = None
        self._script_text = []
        self._pre = None
        self.feed(path.read_text(encoding="utf-8"))
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag in ("script", "style"):
            self._hidden = tag
        if tag == "script":
            self._script_type = attrs.get("type", "").lower()
            self._script_text = []
        if tag == "pre":
            self._pre = []

    def handle_endtag(self, tag):
        if tag == "script" and self._script_type == "application/ld+json":
            self.jsonld.append(json.loads("".join(self._script_text)))
        if tag == self._hidden:
            self._hidden = None
        if tag == "pre" and self._pre is not None:
            self.pre_blocks.append("".join(self._pre))
            self._pre = None

    def handle_data(self, value):
        if self._hidden == "script":
            self._script_text.append(value)
        if not self._hidden:
            self.visible.append(value)
        if self._pre is not None:
            self._pre.append(value)


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def has_type(value, name):
    kind = value.get("@type", [])
    return name == kind or isinstance(kind, list) and name in kind


def papers(value, label):
    records = {}
    for record in objects(value):
        if not has_type(record, "ScholarlyArticle"):
            continue
        identity = record.get("@id")
        assert identity in TITLES, (label, "unexpected paper identity", identity)
        if identity in records:
            assert record == records[identity], (label, "conflicting repeated paper", identity)
        records[identity] = record
    assert set(records) == set(TITLES), (label, "both original paper records are required")
    for identity, record in records.items():
        assert record["name"] == TITLES[identity], (label, "paper title", identity)
        assert [a["name"] for a in record["author"]] == AUTHORS[identity], (label, "paper author order", identity)
    first, final = records[ORIGIN + "/#paper-2016"], records[ORIGIN + "/#paper"]
    assert first["datePublished"] == "2016-09-16" and first["url"] == "https://arxiv.org/abs/1609.04938v1"
    assert final["datePublished"] == "2017" and final["url"] == "https://proceedings.mlr.press/v70/deng17a.html"
    assert final["sameAs"] == "https://arxiv.org/abs/1609.04938v2"
    assert final["isBasedOn"]["@id"] == first["@id"]
    return records


def local_path(url):
    parsed = urlsplit(url)
    target = (SITE / unquote(parsed.path).lstrip("/")).resolve()
    assert target == SITE.resolve() or SITE.resolve() in target.parents, (url, "path leaves site")
    return target / "index.html" if parsed.path.endswith("/") else target


def language_links(page, chinese, english):
    expected = {"en": ORIGIN + english, "zh-Hans": ORIGIN + chinese, "x-default": ORIGIN + english}
    found = [(a.get("hreflang"), a.get("href")) for tag, a in page.tags
             if tag == "link" and "alternate" in a.get("rel", "").split() and "hreflang" in a]
    assert len(found) == len(expected) and dict(found) == expected, (page.path, "language alternates", found)


def check_links(page, url, page_cache):
    for tag, attrs in page.tags:
        for field in ("href", "src"):
            raw = attrs.get(field)
            if not raw:
                continue
            resolved = urlsplit(urljoin(url, raw))
            if (resolved.scheme, resolved.netloc) != ("https", urlsplit(ORIGIN).netloc):
                continue
            if resolved.path in {"/data/" + name for name in DOWNLOADS}:
                assert not resolved.fragment, (page.path, raw, "download cannot have an HTML anchor")
                continue  # Original archives are hosted separately from the source checkout.
            target = local_path(resolved.geturl())
            assert target.is_file(), (page.path, raw, "missing local asset")
            if resolved.fragment:
                assert target.suffix == ".html", (page.path, raw, "anchor target is not HTML")
                if target not in page_cache:
                    page_cache[target] = Page(target)
                ids = {a["id"] for _, a in page_cache[target].tags if "id" in a}
                assert unquote(resolved.fragment) in ids, (page.path, raw, "missing anchor")
        if tag == "img":
            assert attrs.get("alt"), (page.path, "missing image description")


def check_lastmod(value, now):
    assert value, "Every sitemap page needs its supported content-modification date"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        assert dt.date.fromisoformat(value) <= now.date(), (value, "future sitemap date")
    else:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        assert stamp.tzinfo is not None, (value, "timestamp needs a timezone")
        assert stamp <= now, (value, "future sitemap timestamp")


def main():
    cache = {}
    english_dataset = json.loads((SITE / "data/dataset.json").read_text(encoding="utf-8"))
    original_papers = papers(english_dataset, "English dataset")
    bibtex = {}
    for name, expected_hash in BIBTEX_SHA256.items():
        contents = (SITE / name).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == expected_hash, (name, "published citation changed")
        bibtex[name] = contents.decode("utf-8").strip()
        assert not (SITE / "zh" / name).exists(), (name, "use the shared citation file")

    for chinese, english in PAIRS:
        url = ORIGIN + chinese
        page = Page(local_path(url))
        english_page = Page(local_path(ORIGIN + english))
        cache[page.path.resolve()] = page
        assert [a.get("lang") for t, a in page.tags if t == "html"] == ["zh-Hans"], (chinese, "HTML language")
        canonicals = [a.get("href") for t, a in page.tags if t == "link" and "canonical" in a.get("rel", "").split()]
        assert canonicals == [url], (chinese, "self-canonical URL")
        language_links(page, chinese, english)
        language_links(english_page, chinese, english)
        assert sum(t == "h1" for t, _ in page.tags) == 1, (chinese, "exactly one h1 required")
        ids = [a["id"] for _, a in page.tags if "id" in a]
        assert len(ids) == len(set(ids)), (chinese, "duplicate IDs")
        visible = " ".join(page.visible)
        assert re.search(r"[\u3400-\u9fff]", visible), (chinese, "Chinese content missing")
        for title in TITLES.values():
            assert title in visible, (chinese, "original English paper title not visible", title)
        for name in AUTHORS[ORIGIN + "/#paper"]:
            assert name in visible, (chinese, "author name not visible", name)
        assert papers(page.jsonld, chinese) == original_papers, (chinese, "paper records differ from originals")
        assert papers(english_page.jsonld, english) == original_papers, (english, "paper records differ from originals")
        hrefs = {a.get("href") for t, a in page.tags if t == "a"}
        for name in BIBTEX_SHA256:
            assert "/" + name in hrefs or ORIGIN + "/" + name in hrefs, (chinese, "missing shared citation link", name)
        for block in page.pre_blocks:
            if re.match(r"\s*@(?:article|inproceedings)\s*\{", block, re.I):
                assert block.strip() in bibtex.values(), (chinese, "embedded BibTeX differs from shared citation")
        check_links(page, url, cache)

    english_guide = next(obj for obj in objects(Page(SITE / "guide/index.html").jsonld)
                         if has_type(obj, "Article") and obj.get("@id") == ORIGIN + "/guide/#guide")
    chinese_guide = next(obj for obj in objects(Page(SITE / "zh/guide/index.html").jsonld)
                         if has_type(obj, "Article") and obj.get("@id") == ORIGIN + "/zh/guide/#guide")
    assert english_guide["workTranslation"]["@id"] == chinese_guide["@id"], "English translation link must identify the Chinese article"
    assert chinese_guide["translationOfWork"]["@id"] == english_guide["@id"], "Chinese article must identify its English original"

    chinese_dataset = json.loads((SITE / "zh/data/dataset.json").read_text(encoding="utf-8"))
    for key in ("@id", "@type", "url", "name", "creator", "citation", "distribution", "license", "sameAs", "datePublished"):
        assert chinese_dataset[key] == english_dataset[key], ("Chinese dataset changed original field", key)
    assert chinese_dataset["description"] != english_dataset["description"]
    assert re.search(r"[\u3400-\u9fff]", chinese_dataset["description"]), "Dataset description must be Chinese"
    assert papers(chinese_dataset.get("subjectOf"), "Chinese dataset subjectOf") == original_papers
    assert papers(chinese_dataset, "Chinese dataset") == original_papers
    embedded = [obj for obj in objects(cache[local_path(ORIGIN + "/zh/data/")].jsonld)
                if has_type(obj, "Dataset") and obj.get("@id") == english_dataset["@id"]]
    assert embedded and all(obj == chinese_dataset for obj in embedded), "Chinese dataset HTML and JSON must agree"
    manifest = json.loads((SITE / "data/files.json").read_text(encoding="utf-8"))["files"]
    assert len(manifest) == 12 and {item["file"] for item in manifest} == DOWNLOADS
    links = {a.get("href") for t, a in cache[local_path(ORIGIN + "/zh/data/")].tags if t == "a"}
    assert {"/data/" + name for name in DOWNLOADS} <= links, "Chinese downloads must keep all twelve /data/ URLs"
    assert chinese_dataset["distribution"] == english_dataset["distribution"], "Preserve the original dataset distributions; the separate handwriting archive remains a related download"

    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    entries = ET.parse(SITE / "sitemap.xml").getroot().findall("s:url", namespace)
    urls = [entry.findtext("s:loc", namespaces=namespace) for entry in entries]
    expected = {ORIGIN + path for pair in PAIRS for path in pair} | {ORIGIN + "/demo.html"}
    assert len(urls) == 7 and set(urls) == expected, ("sitemap canonical URL set", urls)
    now = dt.datetime.now(dt.timezone.utc)
    for entry in entries:
        check_lastmod(entry.findtext("s:lastmod", namespaces=namespace), now)
    print("Validated three Chinese pages, language links, original attribution, shared citations, twelve downloads and seven sitemap URLs.")


if __name__ == "__main__":
    main()
