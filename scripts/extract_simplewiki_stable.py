"""Stable sentences from Simple English Wikipedia's revision history (CC BY-SA):
text editors kept seeing and left alone.

usage: bzip2 -dc simplewiki-latest-pages-meta-history.xml.bz2 | python3 -I extract_simplewiki_stable.py <out_dir> [min revisions = 20] [min days = 365]

For each article, every line of wikitext remembers the revision and date since
which it has stood unchanged. Lines of the final revision that survived at
least `min revisions` later revisions and `min days` are cleaned (markup
stripped as in extract_corpus.py) and split into sentences. Vandalized
revisions don't reset a line's age if a revert restores it (reverted
revisions are skipped). Writes stable_{train,dev,test}.tsv, split by article
as extract_simplewiki_edits.py does: page id, revisions survived, days, sentence.
"""
import collections, datetime, re, sys, xml.etree.ElementTree as ET

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from extract_simplewiki_edits import REVERT, ok, sentences  # noqa: E402

out = sys.argv[1]
min_revs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
min_days = int(sys.argv[3]) if len(sys.argv) > 3 else 365
files = {k: open(f"{out}/stable_{k}.tsv", "w") for k in ("train", "dev", "test")}
stats = collections.Counter()
page_id = page_ns = None
revs = []  # (sha1, comment, timestamp, text)


def when(ts):
    return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def flush():
    if page_ns != "0" or not revs:
        return
    # Drop reverted revisions (and the reverts) so vandalism doesn't reset ages.
    first_seen, bad = {}, set()
    for i, (sha, comment, _, _) in enumerate(revs):
        if sha in first_seen:
            bad.update(range(first_seen[sha] + 1, i + 1))
        else:
            first_seen[sha] = i
        if REVERT.search(comment or ""):
            bad.add(i)
    kept = [r for i, r in enumerate(revs) if i not in bad]
    if not kept:
        return  # every revision was vandalism or a revert
    since = {}  # line -> (revision index, date) since which it stood unchanged
    for i, (_, _, ts, text) in enumerate(kept):
        lines = set(text.split("\n"))
        since = {ln: since.get(ln, (i, ts)) for ln in lines}
    last, end = len(kept) - 1, when(kept[-1][2])
    split = "test" if int(page_id) % 20 == 0 else "dev" if int(page_id) % 20 == 1 else "train"
    for ln, (i, ts) in since.items():
        survived, days = last - i, (end - when(ts)).days
        if survived < min_revs or days < min_days:
            continue
        for s in sentences(ln):
            if ok(s):
                files[split].write(f"{page_id}\t{survived}\t{days}\t{s}\n")
                stats[f"stable {split}"] += 1


for ev, el in ET.iterparse(sys.stdin.buffer, events=("start", "end")):
    tag = el.tag.rsplit("}", 1)[-1]
    if ev == "start":
        if tag == "page":
            revs, page_id, page_ns = [], None, None
        continue
    if tag == "ns" and page_ns is None:
        page_ns = el.text
    elif tag == "id" and page_id is None:
        page_id = el.text
    elif tag == "revision":
        g = lambda name: next((c.text for c in el if c.tag.endswith(name)), None)
        if page_ns == "0":
            revs.append((g("sha1") or "", g("comment") or "", g("timestamp") or "2001-01-01T00:00:00Z", g("text") or ""))
        el.clear()
    elif tag == "page":
        flush()
        stats["articles"] += page_ns == "0"
        if stats["articles"] % 20000 == 0 and page_ns == "0":
            print(dict(stats), file=sys.stderr, flush=True)
        el.clear()
for f in files.values():
    f.close()
print(dict(stats))
