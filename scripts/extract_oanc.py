"""Plain sentences from the OANC's written genres (Open American National
Corpus; unrestricted, commercial use allowed), with any sentence sharing an
8-word stretch with a SWORDS dev or test passage removed, so SWORDS stays a
clean test (its passages come from MASC, drawn from the same genres).

usage: python3 -I extract_oanc.py <OANC-GrAF/data> <swords dir> <out.txt>
"""
import glob, gzip, json, os, re, sys

data, swords, out = sys.argv[1:4]
words = lambda s: re.findall(r"[a-z0-9']+", s.lower())
shingles = set()
for split in ("dev", "test"):
    d = json.load(open(os.path.join(swords, f"swords-v1.1_{split}.json")))
    for c in d["contexts"].values():
        w = words(c["context"])
        shingles.update(tuple(w[i:i + 8]) for i in range(len(w) - 7))
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
GOOD = re.compile(r"^[A-Z0-9\"'(].*[.!?\"')]$")
kept = dropped = 0
by_genre = {}
with open(out, "w") as o:
    for path in sorted(glob.glob(os.path.join(data, "written_*", "**", "*.txt"), recursive=True)):
        genre = path.split(os.sep)[-3] if "written_" in path.split(os.sep)[-3] else path.split(os.sep)[len(data.split(os.sep)) + 1]
        text = open(path, encoding="utf-8", errors="replace").read()
        for para in re.split(r"\n\s*\n", text):
            for s in SPLIT.split(" ".join(para.split())):
                n = len(s.split())
                if not (4 <= n <= 40) or not GOOD.match(s) or re.search(r"[<>{}\[\]|=#*_@]|https?:", s):
                    continue
                w = words(s)
                if any(tuple(w[i:i + 8]) in shingles for i in range(len(w) - 7)):
                    dropped += 1
                    continue
                o.write(s + "\n"); kept += 1
                by_genre[genre] = by_genre.get(genre, 0) + n
print(f"kept {kept} sentences; dropped {dropped} overlapping SWORDS passages")
for g, n in sorted(by_genre.items(), key=lambda x: -x[1]): print(f"  {g}: {n} words")
