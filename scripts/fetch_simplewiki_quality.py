"""Simple English Wikipedia's editor-certified articles (Good articles, Very
good articles; CC BY-SA), as plain-text paragraphs: one document per line.

usage: python3 -I fetch_simplewiki_quality.py <out_dir>
Writes paragraphs.tsv (title, category, paragraph) and titles.txt (credits).
"""
import json, sys, time, urllib.parse, urllib.request

out = sys.argv[1]
API = "https://simple.wikipedia.org/w/api.php"
UA = {"User-Agent": "Scriberocks-research/1.0 (https://scribe.rocks)"}


def get(**params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


rows, titles = [], []
for cat in ["Very good articles", "Good articles"]:
    members = get(action="query", list="categorymembers", cmtitle=f"Category:{cat}", cmlimit=500, cmnamespace=0)["query"]["categorymembers"]
    for m in members:
        page = next(iter(get(action="query", prop="extracts", explaintext=1, exsectionformat="plain", titles=m["title"])["query"]["pages"].values()))
        text = page.get("extract", "")
        titles.append(f"{m['title']}\t{cat}")
        for para in text.split("\n"):
            para = " ".join(para.split())
            # Prose paragraphs only: headings and list stubs are short or unpunctuated.
            if len(para.split()) >= 15 and para[-1:] in ".!?\"')":
                rows.append((m["title"], cat, para))
        time.sleep(0.3)
with open(f"{out}/paragraphs.tsv", "w") as f:
    for t, c, p in rows:
        f.write(f"{t}\t{c}\t{p}\n")
with open(f"{out}/titles.txt", "w") as f:
    f.write("\n".join(titles) + "\n")
print(f"{len(titles)} articles, {len(rows)} paragraphs")
