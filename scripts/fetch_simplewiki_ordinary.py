"""Ordinary (uncertified) Simple English Wikipedia articles as plain-text
paragraphs (CC BY-SA), the comparison for editor-certified ones.

usage: python3 -I fetch_simplewiki_ordinary.py <out_dir> [articles = 400]
Writes ordinary.tsv (title, paragraph); skips Good / Very good articles.
"""
import json, sys, time, urllib.parse, urllib.request

out = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else 400
API = "https://simple.wikipedia.org/w/api.php"
UA = {"User-Agent": "Scriberocks-research/1.0 (https://scribe.rocks)"}


def get(**params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


certified = {l.split("\t")[0] for l in open(f"{out}/titles.txt")}
rows, seen = [], set()
while len(seen) < want:
    for page in get(action="query", list="random", rnnamespace=0, rnlimit=20)["query"]["random"]:
        t = page["title"]
        if t in seen or t in certified:
            continue
        seen.add(t)
        p = next(iter(get(action="query", prop="extracts", explaintext=1, exsectionformat="plain", titles=t)["query"]["pages"].values()))
        for para in p.get("extract", "").split("\n"):
            para = " ".join(para.split())
            if len(para.split()) >= 15 and para[-1:] in ".!?\"')":
                rows.append((t, para))
        time.sleep(0.3)
    time.sleep(0.5)
with open(f"{out}/ordinary.tsv", "w") as f:
    for t, p in rows:
        f.write(f"{t}\t{p}\n")
print(f"{len(seen)} ordinary articles, {len(rows)} paragraphs")
