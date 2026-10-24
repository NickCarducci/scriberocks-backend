"""Sentence edits from Simple English Wikipedia's revision history (CC BY-SA).

usage: bzip2 -dc simplewiki-latest-pages-meta-history.xml.bz2 | python3 -I extract_simplewiki_edits.py <out_dir>

For each article, consecutive revisions are compared paragraph by paragraph;
changed paragraphs are cleaned with the same markup stripping as the corpus
(extract_corpus.py: link text kept), split into sentences, and edited
sentences paired. Kept: small edits (1-6 words changed, 4-60 words long) that
change no name or number and leave a well-formed sentence. Dropped:
vandalism (any revision a later one reverts, by checksum) and the reverts.
Writes train.tsv, dev.tsv, test.tsv (split by article): page id, before, after.
"""
import collections, difflib, re, sys, xml.etree.ElementTree as ET

out = sys.argv[1]
GOOD = re.compile(r"^[A-Z0-9\"'(].*[.!?\"')]$")
BAD = re.compile(r"[<>{}\[\]|=#*_]|https?:")
REVERT = re.compile(r"\b(revert|reverted|rv|rvv|undo|undid|vandal)", re.I)
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
ABBR = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|e\.g|i\.e|U\.S|No)\.$")
TOKEN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


def strip(t):
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref.*?</ref>", "", t, flags=re.S)
    for _ in range(5):
        t2 = re.sub(r"\{\{[^{}]*\}\}", "", t)
        if t2 == t: break
        t = t2
    t = re.sub(r"\[\[(?:File|Image|Category|[a-z\-]+):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    return t.replace("&nbsp;", " ").replace("&amp;", "&")


def sentences(para):
    para = strip(para).strip()
    if not para or para[0] in "=*#:;|!{": return []
    parts, buf = [], ""
    for piece in SPLIT.split(para):
        buf = f"{buf} {piece}".strip() if buf else piece
        if not ABBR.search(buf): parts.append(buf); buf = ""
    if buf: parts.append(buf)
    return [" ".join(s.split()) for s in parts]


def ok(s):
    n = len(s.split())
    return 4 <= n <= 60 and GOOD.match(s) and not BAD.search(s)


def facty(i, w):
    return any(c.isdigit() for c in w) or (i > 0 and w[:1].isupper())


def small_reword(a, b):
    """1-6 tokens changed, no name or number among them."""
    ta, tb = TOKEN.findall(a), TOKEN.findall(b)
    changed = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op == "equal": continue
        changed += (i2 - i1) + (j2 - j1)
        if any(facty(i, ta[i]) for i in range(i1, i2)) or any(facty(j, tb[j]) for j in range(j1, j2)): return False
    return 1 <= changed <= 6


def edits(prev, cur):
    """Edited sentence pairs between two revisions' wikitext."""
    pl, cl = prev.split("\n"), cur.split("\n")
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pl, cl, autojunk=False).get_opcodes():
        if op != "replace" or (i2 - i1) != (j2 - j1): continue
        for a, b in zip(pl[i1:i2], cl[j1:j2]):
            sa, sb = sentences(a), sentences(b)
            for o2, x1, x2, y1, y2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
                if o2 == "replace" and (x2 - x1) == (y2 - y1):
                    for s, t in zip(sa[x1:x2], sb[y1:y2]):
                        if s != t and ok(s) and ok(t) and small_reword(s, t): out.append((s, t))
    return out


files = {k: open(f"{out}/{k}.tsv", "w") for k in ("train", "dev", "test")}
seen = set()
stats = collections.Counter()
ns = None
page_id = page_ns = None
revs = []  # (sha1, comment, text) for the current page


def flush():
    """Pairs from one article, leaving out reverted revisions and the reverts."""
    if page_ns != "0" or len(revs) < 2: return
    first_seen, bad = {}, set()
    for i, (sha, comment, _) in enumerate(revs):
        if sha in first_seen:
            bad.update(range(first_seen[sha] + 1, i + 1))  # reverted revisions, and the revert
        else:
            first_seen[sha] = i
        if REVERT.search(comment or ""): bad.add(i)
    split = "test" if int(page_id) % 20 == 0 else "dev" if int(page_id) % 20 == 1 else "train"
    for i in range(1, len(revs)):
        stats["revisions"] += 1
        if i in bad or (i - 1) in bad:
            stats["skipped (vandalism or revert)"] += 1
            continue
        for s, t in edits(revs[i - 1][2], revs[i][2]):
            if (s, t) in seen: continue
            seen.add((s, t))
            files[split].write(f"{page_id}\t{s}\t{t}\n")
            stats[f"pairs {split}"] += 1


for ev, el in ET.iterparse(sys.stdin.buffer, events=("start", "end")):
    tag = el.tag.rsplit("}", 1)[-1]
    if ev == "start":
        if tag == "page": revs, page_id, page_ns = [], None, None
        continue
    if tag == "ns" and page_ns is None: page_ns = el.text
    elif tag == "id" and page_id is None: page_id = el.text
    elif tag == "revision":
        g = lambda name: next((c.text for c in el if c.tag.endswith(name)), None)
        if page_ns == "0":
            revs.append((g("sha1") or "", g("comment") or "", g("text") or ""))
        el.clear()
    elif tag == "page":
        flush()
        stats["articles"] += page_ns == "0"
        if stats["articles"] % 5000 == 0 and page_ns == "0":
            print(dict(stats), file=sys.stderr, flush=True)
        el.clear()
for f in files.values(): f.close()
print(dict(stats))
