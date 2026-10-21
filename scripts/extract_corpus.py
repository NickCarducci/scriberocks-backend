"""Plain sentences from Tatoeba (English export) and a Simple English Wikipedia dump.

usage: python3 -I extract.py <tatoeba.tsv.bz2> <simplewiki.xml.bz2> <out_dir>
Writes tatoeba.txt, tatoeba_authors.txt (for credits), simplewiki.txt: one sentence per line.
"""
import bz2, re, sys, collections, xml.etree.ElementTree as ET

tat, wiki, out = sys.argv[1:4]
GOOD = re.compile(r"^[A-Z0-9\"'(].*[.!?\"')]$")

def keep(s, lo=3, hi=40):
    n = len(s.split())
    return lo <= n <= hi and GOOD.match(s) and not re.search(r"[<>{}\[\]|=#*_]|https?:", s)

# Tatoeba
authors = collections.Counter()
n = 0
with bz2.open(tat, "rt", encoding="utf-8") as f, open(f"{out}/tatoeba.txt", "w") as o:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4 or p[1] != "eng":
            continue
        s = " ".join(p[2].split())
        if keep(s, 2):
            o.write(s + "\n"); n += 1
            if p[3] != "\\N": authors[p[3]] += 1
with open(f"{out}/tatoeba_authors.txt", "w") as o:
    for a, c in authors.most_common():
        o.write(f"{a}\t{c}\n")
print(f"tatoeba: {n} sentences, {len(authors)} authors")

# Simple English Wikipedia: main-namespace articles, markup stripped.
def strip(t):
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref.*?</ref>", "", t, flags=re.S)
    for _ in range(5):  # nested templates / tables
        t2 = re.sub(r"\{\{[^{}]*\}\}", "", t)
        t2 = re.sub(r"\{\|[^{}]*?\|\}", "", t2, flags=re.S)
        if t2 == t: break
        t = t2
    t = re.sub(r"\[\[(?:File|Image|Category|[a-z\-]+):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("&nbsp;", " ").replace("&amp;", "&")
    return t

SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
ABBR = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|e\.g|i\.e|U\.S|No)\.$")
n = articles = 0
ns = "{http://www.mediawiki.org/xml/export-0.11/}"
with bz2.open(wiki, "rb") as f, open(f"{out}/simplewiki.txt", "w") as o:
    for ev, el in ET.iterparse(f, events=("end",)):
        if not el.tag.endswith("page"): continue
        tag = el.tag[:-4]
        if el.findtext(f"{tag}ns") == "0" and el.find(f"{tag}redirect") is None:
            text = el.findtext(f"{tag}revision/{tag}text") or ""
            articles += 1
            for para in strip(text).split("\n"):
                para = para.strip()
                if not para or para[0] in "=*#:;|!{": continue
                parts, buf = [], ""
                for piece in SPLIT.split(para):
                    buf = f"{buf} {piece}".strip() if buf else piece
                    if not ABBR.search(buf): parts.append(buf); buf = ""
                if buf: parts.append(buf)
                for s in parts:
                    s = " ".join(s.split())
                    if keep(s, 4): o.write(s + "\n"); n += 1
        el.clear()
print(f"simplewiki: {articles} articles, {n} sentences")
