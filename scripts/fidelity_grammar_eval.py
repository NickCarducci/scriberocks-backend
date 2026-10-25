"""Can FidelityAI catch grammar? Editors' corrections from Simple English
Wikipedia history (examples/grammar_edits.rs): run-on splits, inflection
(agreement, tense, number) and function-word fixes. Real edits only; the
rulebook is 5,000 stable sentences (proper use), local and free.

For each correction, the after should score more typical than the before.
Reported per kind: share of edits where it does (0.5 chance), for
  prose   FidelityAI on the text (path B grammar frames)
  tags    FidelityAI on the fine part-of-speech tag sequence (path D)
  shape   FidelityAI on per-sentence shape records (words, commas, joining
          words, verbs; path A); a text scores as its least typical sentence
  tag trigrams  plain count baseline: mean log-probability of the tag
          trigrams under the rulebook's counts
Writes before/after pairs for the judge (text_judge_eval: group, label, text).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_grammar_eval.py <grammar dir> <engine dir> [per kind = 600]
"""
import collections, json, math, random, sys, types

gdir, engine = sys.argv[1], sys.argv[2]
per_kind = int(sys.argv[3]) if len(sys.argv) > 3 else 600
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rulebook = [json.loads(l) for l in open(f"{gdir}/rulebook.jsonl")]
edits = collections.defaultdict(list)
for l in open(f"{gdir}/edits.jsonl"):
    e = json.loads(l)
    edits[e["kind"]].append(e)
rng = random.Random(9)
pairs = []
for kind, es in sorted(edits.items()):
    rng.shuffle(es)
    pairs += es[:per_kind]
print("edits: " + ", ".join(f"{k} {min(len(v), per_kind)}" for k, v in sorted(edits.items())) + f"; rulebook {len(rulebook)} stable sentences")
texts = [p[side] for p in pairs for side in ("before", "after")]  # before, after, before, after, ...
with open(f"{gdir}/judge_pairs.tsv", "w") as f:
    for p in pairs:
        for y, side in ((0, "before"), (1, "after")):
            f.write(f"{p['id']}\t{y}\t{p[side]['text']}\n")


def report(name, typicality):
    """typicality per text (higher = more typical), aligned with texts."""
    by = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        b, a = typicality[2 * i], typicality[2 * i + 1]
        if a is not None and b is not None:
            by[p["kind"]].append((a > b) + 0.5 * (a == b))
    print(f"  {name}: " + ", ".join(f"{k} {sum(v) / len(v):.3f} ({len(v)})" for k, v in sorted(by.items())), flush=True)


tag_ids = {}
tid = lambda t: tag_ids.setdefault(t, len(tag_ids))
seq = lambda d: [tid(t) for s in d["sentences"] for t in ["<s>"] + s["xpos"]]
for d in rulebook + texts:
    seq(d)  # the tag inventory is closed (Penn Treebank): fix ids up front

# Plain count baseline: tag trigrams.
tri = collections.Counter(tuple(x[i:i + 3]) for x in map(seq, rulebook) for i in range(len(x) - 2))
total, types_ = sum(tri.values()), len(tri) + 1
def trigram_lp(d):
    x = seq(d)
    grams = [tuple(x[i:i + 3]) for i in range(len(x) - 2)]
    return sum(math.log((tri[g] + 1) / (total + types_)) for g in grams) / max(len(grams), 1)
report("tag trigrams (count)", [trigram_lp(d) for d in texts])


def fit(source, cands, label):
    return abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=source, label=label),
                              artifact_dataset=types.SimpleNamespace(episodes=cands, label="candidates"), max_qubits=20, seed=42)


# Prose (path B): a text scores as the mean over its frames.
res = fit([d["text"] for d in rulebook], [d["text"] for d in texts], "stable")
per = collections.defaultdict(list)
for s, p in zip(res["artifact_scores"], res.get("artifact_provenance") or []):
    if s is not None and isinstance(p, dict) and "sourceIndex" in p:
        per[p["sourceIndex"]].append(-s)
report(f"prose (path {res['effective_path']}, k={res['selected'].get('k')})", [sum(per[i]) / len(per[i]) if per.get(i) else None for i in range(len(texts))])

# Tags (path D): one integer sequence per text.
res = fit([[{"sequence": seq(d)}] for d in rulebook], [[{"sequence": seq(d)}] for d in texts], "stable-tags")
report(f"tags (path {res['effective_path']}, k={res['selected'].get('k')})", [None if s is None else -s for s in res["artifact_scores"]])

# Shape (path A): one record per sentence; a text scores as its least typical sentence.
bucket = lambda n: str(min(n // 5, 8))
shape = lambda s: {"words": bucket(s["shape"]["words"]), "commas": str(min(s["shape"]["commas"], 4)),
                   "joiners": str(min(s["shape"]["joiners"], 3)), "verbs": str(min(s["shape"]["verbs"], 5))}
src = [shape(s) for d in rulebook for s in d["sentences"]]
cands, owner = [], []
for i, d in enumerate(texts):
    for s in d["sentences"]:
        cands.append(shape(s)); owner.append(i)
res = fit(src, cands, "stable-shape")
worst = collections.defaultdict(list)
for i, s in zip(owner, res["artifact_scores"]):
    if s is not None:
        worst[i].append(s)
report(f"shape (path {res['effective_path']}, k={res['selected'].get('k')})", [-max(worst[i]) if worst.get(i) else None for i in range(len(texts))])
