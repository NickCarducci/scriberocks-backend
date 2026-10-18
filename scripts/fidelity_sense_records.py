"""Sense-stage training data for FidelityAI's role-embedding encoding.

Records are (nearest content word before, SENSE, nearest content word after)
for ambiguous words whose senses offer synonym options (the editor's case),
from SemCor's even (training) documents; tests are every such occurrence in
the odd (held-out) documents.

  python3 -I scripts/fidelity_sense_records.py data [max-records] [per-sense] [stop-words] [layout]

layout "sense" (default): (before, SENSE, after), k=3.
layout "lemma_rank": (before, LEMMA, RANK, after), k=4 — the sense as which word
(learned embedding) and which of its option-bearing senses (one-hot rank, 0–7),
so a word's senses stay exactly distinct without a one-hot over every sense.

Writes data/fidelity/senses[_lemma_rank]/: source.jsonl, tests.jsonl, vocab.json.
"""
import glob, html, json, os, random, re, sys
from collections import Counter, defaultdict

data = sys.argv[1]
max_records, per_sense, n_stop = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((2, 5000), (3, 8), (4, 100)))
layout = sys.argv[5] if len(sys.argv) > 5 else "sense"
MAX_RANK = 8
oewn, sem = f"{data}/oewn", f"{data}/semcor/wsdeval/WSD_Evaluation_Framework/Training_Corpora/SemCor"
out = f"{data}/fidelity/senses" + ("_lemma_rank" if layout == "lemma_rank" else "")

# WordNet: sense key → synset; lemma senses in WordNet order; synset members.
key_syn, senses_of = {}, defaultdict(list)
for path in glob.glob(f"{oewn}/entries-*.json"):
    for lemma, by_pos in json.load(open(path)).items():
        for pos, entry in by_pos.items():
            for s in entry.get("sense", []):
                key_syn[s["id"]] = s["synset"]
                senses_of[(lemma.lower(), pos[0])].append(s["synset"])
members = {}
for path in glob.glob(f"{oewn}/*.json"):
    name = os.path.basename(path)
    if name.split(".")[0] in ("noun", "verb", "adj", "adv"):
        for sid, syn in json.load(open(path)).items():
            members[sid] = [m.lower() for m in syn.get("members", [])]
POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
def options(synset, lemma):
    return [m for m in members.get(synset, []) if m != lemma and " " not in m and m.isalpha() and len(m) > 1]

# SemCor sentences with gold keys.
gold = {}
for line in open(f"{sem}/semcor.gold.key.txt"):
    p = line.split(); gold[p[0]] = p[1:]
docs, sents = [], []
doc = None
for line in open(f"{sem}/semcor.data.xml"):
    line = line.strip()
    if line.startswith("<text "): doc = re.search(r'id="([^"]+)"', line).group(1); docs.append(doc)
    elif line.startswith("<sentence"): words, tags = [], []
    elif line.startswith("</sentence>"): sents.append((doc, words, tags))
    else:
        m = re.match(r'<(wf|instance)([^>]*)>(.*)</', line)
        if not m: continue
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
        if m.group(1) == "instance" and a.get("pos") in POS and a["id"] in gold:
            tags.append((len(words), a["lemma"].lower(), POS[a["pos"]], gold[a["id"]]))
        words.append(html.unescape(m.group(3)).lower())
held = set(sorted(set(docs))[1::2])
stop = {w for w, _ in Counter(w for d, ws, _ in sents if d not in held for w in ws).most_common(n_stop)}
def ctx(words, i):
    left = next((words[j] for j in range(i - 1, -1, -1) if words[j] not in stop and any(c.isalpha() for c in words[j])), "<s>")
    right = next((words[j] for j in range(i + 1, len(words)) if words[j] not in stop and any(c.isalpha() for c in words[j])), "</s>")
    return left, right

def cases(train):
    for d, words, tags in sents:
        if (d in held) == train: continue
        for i, lemma, pos, keys in tags:
            syns = [key_syn[k] for k in keys if k in key_syn]
            cands = [s for s in senses_of.get((lemma, pos), []) if options(s, lemma)]
            if len(cands) < 2 or not syns: continue
            yield {"lemma": lemma, "pos": pos, "gold": syns, "candidates": cands, "context": ctx(words, i)}

train = list(cases(True))
random.Random(7).shuffle(train)
by_sense, picked = Counter(), []
for c in train:
    g = c["gold"][0]
    if g in c["candidates"] and by_sense[g] < per_sense:
        by_sense[g] += 1; picked.append(c)
    if len(picked) >= max_records: break
# Dense vocabulary: 0 boundary, 1 unknown, then context words, then lemma or
# sense tokens, then (lemma_rank) the rank tokens.
words_v = sorted({w for c in picked for w in c["context"]} - {"<s>", "</s>"})
senses_v = sorted({c["gold"][0] for c in picked})
lemmas_v = sorted({f"{c['lemma']}.{c['pos']}" for c in picked})
vocab = {"<boundary>": 0, "<unknown>": 1, **{w: 2 + i for i, w in enumerate(words_v)}}
sense_id = {s: 2 + len(words_v) + i for i, s in enumerate(senses_v)}
lemma_id = {l: 2 + len(words_v) + i for i, l in enumerate(lemmas_v)}
rank_id = {r: 2 + len(words_v) + len(lemmas_v) + r for r in range(MAX_RANK)}
enc = lambda w: 0 if w in ("<s>", "</s>") else vocab.get(w, 1)
def record(c, sense):
    before, after = enc(c["context"][0]), enc(c["context"][1])
    if layout == "lemma_rank":
        rank = min(c["candidates"].index(sense), MAX_RANK - 1)
        return [before, lemma_id.get(f"{c['lemma']}.{c['pos']}", 1), rank_id[rank], after]
    return [before, sense_id[sense], after]
os.makedirs(out, exist_ok=True)
with open(f"{out}/source.jsonl", "w") as f:
    for c in picked:
        f.write(json.dumps({"sequence": record(c, c["gold"][0])}) + "\n")
tests = list(cases(False))
with open(f"{out}/tests.jsonl", "w") as f:
    for c in tests:
        cand = c["candidates"][:MAX_RANK] if layout == "lemma_rank" else c["candidates"]
        f.write(json.dumps({"lemma": c["lemma"], "pos": c["pos"], "gold": c["gold"], "candidates": cand,
                            "records": [record(c, s) for s in cand] if layout == "lemma_rank" or all(s in sense_id for s in cand) else None,
                            "context": [enc(c["context"][0]), enc(c["context"][1])]}) + "\n")
json.dump({"layout": layout, "words": vocab, "senses": sense_id, "lemmas": lemma_id, "ranks": rank_id}, open(f"{out}/vocab.json", "w"))
lemmas = {(c["lemma"], c["pos"]) for c in picked}
print(f"{len(picked)} training records ({len(senses_v)} senses, {len(lemmas)} words, vocabulary {2 + len(words_v) + len(senses_v)}); "
      f"{len(tests)} held-out test cases; stop list {n_stop}")
