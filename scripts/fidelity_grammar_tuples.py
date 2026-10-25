"""FidelityAI's per-tuple scores (role embeddings: each tuple scored on its
own against the source's tuples, fitter.tuple_scores) on editors' grammar
corrections. Each sentence's fine part-of-speech tags are given as sliding
trigram windows, so every trigram is a tuple (k = 3). Rulebook: 5,000 stable
Simple English Wikipedia sentences. Local and free.

  detect  the after's worst tuple is more typical than the before's (0.5 chance)
  locate  the before's worst tuple covers the edited word (chance: the share
          of windows that cover it)
Baseline: plain tag-trigram counts in the rulebook (the rarest trigram is the
worst), for both.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_grammar_tuples.py <grammar dir> <engine dir> [per kind = 600]
"""
import collections, json, random, sys, types

import numpy as np

gdir, engine = sys.argv[1], sys.argv[2]
per_kind = int(sys.argv[3]) if len(sys.argv) > 3 else 600
K = 3
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

tag_ids = {}
tid = lambda t: tag_ids.setdefault(t, len(tag_ids))
flat = lambda d, key: [x for s in d["sentences"] for x in ["<s>"] + s[key]]
stream = lambda d: [tid(t) for t in flat(d, "xpos")]
windows = lambda d: [stream(d)[i:i + K] for i in range(len(stream(d)) - K + 1)]
as_seq = lambda d: [t for w in windows(d) for t in w]  # chunks of K = the sliding windows
texts = [p[s] for p in pairs for s in ("before", "after")]
for d in rulebook + texts:
    stream(d)


def edit_index(p):
    b, a = [w.lower() for w in flat(p["before"], "tokens")], [w.lower() for w in flat(p["after"], "tokens")]
    i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return min(i, len(b) - 1)


counts = collections.Counter(tuple(w) for d in rulebook for w in windows(d))
count_worst = lambda d: [-counts[tuple(w)] for w in windows(d)]  # higher = worse

res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[[{"sequence": as_seq(d)}] for d in rulebook], label="stable-tags"),
                         artifact_dataset=types.SimpleNamespace(episodes=[[{"sequence": as_seq(d)}] for d in texts], label="candidates"),
                         max_qubits=20, seed=42, encoding={"type": "role_embedding_v1", "dim": 16, "one_hot_roles": []}, k_values=[K])
pipe = res["pipe"]
assert int(pipe.grammar.tuple_size) == K and len(res["mapped_artifacts"]) == len(texts)
fid = pipe.tuple_scores(res["mapped_artifacts"])
print(f"fit: path {res['effective_path']}, k={K}, features {pipe._T.shape[0]}; {len(texts)} texts", flush=True)

stats = collections.defaultdict(lambda: collections.defaultdict(list))
for i, p in enumerate(pairs):
    e = edit_index(p)
    for name, before, after in (("FidelityAI", fid[2 * i], fid[2 * i + 1]), ("trigram counts", count_worst(p["before"]), count_worst(p["after"]))):
        if not before or not after:
            continue
        s = stats[p["kind"]]
        s[f"{name} detect"].append((max(after) < max(before)) + 0.5 * (max(after) == max(before)))
        worst = [j for j, v in enumerate(before) if v == max(before)]
        covering = {j for j in range(len(before)) if j <= e <= j + K - 1}
        s[f"{name} locate"].append(len(covering & set(worst)) / len(worst))
        if name == "FidelityAI":
            s["chance locate"].append(len(covering) / len(before))

m = lambda xs: sum(xs) / max(len(xs), 1)
for kind in sorted(stats):
    s = stats[kind]
    print(f"{kind} ({len(s['FidelityAI detect'])} edits): detect — FidelityAI {m(s['FidelityAI detect']):.3f}, counts {m(s['trigram counts detect']):.3f}; "
          f"locate — FidelityAI {m(s['FidelityAI locate']):.3f}, counts {m(s['trigram counts locate']):.3f}, chance {m(s['chance locate']):.3f}")
