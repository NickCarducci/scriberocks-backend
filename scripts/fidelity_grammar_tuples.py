"""FidelityAI's per-tuple scores (role embeddings: each tuple scored on its
own against the source's tuples, fitter.tuple_scores) on editors' grammar
corrections. Each sentence's fine part-of-speech tags are given as sliding
windows of k tags, so every window is a tuple (default k = 3). Rulebook: 5,000 stable
Simple English Wikipedia sentences. Local and free.

  detect  the after's worst tuple is more typical than the before's (0.5 chance)
  locate  the before's worst tuple covers the edited word (chance: the share
          of windows that cover it)
Baseline: plain tag-window counts in the rulebook (the rarest window is the
worst), for both.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_grammar_tuples.py <grammar dir> <engine dir> [per kind = 600] [k = 3] [embedding dim = 16] [unique]
(features per record: dim^k, so a longer window needs a smaller dim)
unique: each tag window enters the rulebook once (its first occurrence), so
common patterns don't outweigh rare ones in the fit; counts stay as they are.
"""
import collections, json, random, sys, types

import numpy as np

gdir, engine = sys.argv[1], sys.argv[2]
per_kind = int(sys.argv[3]) if len(sys.argv) > 3 else 600
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3
DIM = int(sys.argv[5]) if len(sys.argv) > 5 else 16
UNIQUE = len(sys.argv) > 6 and sys.argv[6] == "unique"
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
for d in rulebook + [p[s] for p in pairs for s in ("before", "after")]:
    stream(d)
# Edits whose before and after both have a full window (k tags or more).
pairs = [p for p in pairs if windows(p["before"]) and windows(p["after"])]
texts = [p[s] for p in pairs for s in ("before", "after")]


def edit_index(p):
    b, a = [w.lower() for w in flat(p["before"], "tokens")], [w.lower() for w in flat(p["after"], "tokens")]
    i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return min(i, len(b) - 1)


counts = collections.Counter(tuple(w) for d in rulebook for w in windows(d))
count_worst = lambda d: [-counts[tuple(w)] for w in windows(d)]  # higher = worse

seen_windows = set()


def rulebook_seq(d):
    if not UNIQUE:
        return as_seq(d)
    fresh = [w for w in windows(d) if tuple(w) not in seen_windows]
    seen_windows.update(map(tuple, fresh))
    return [t for w in fresh for t in w]


source = [s for s in map(rulebook_seq, rulebook) if s]
print(f"rulebook: {len(source)} sentences, {sum(len(s) for s in source) // K} windows" + (" (each window once)" if UNIQUE else ""), flush=True)
res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[[{"sequence": s}] for s in source], label="stable-tags"),
                         artifact_dataset=types.SimpleNamespace(episodes=[[{"sequence": as_seq(d)}] for d in texts], label="candidates"),
                         max_qubits=20, seed=42, encoding={"type": "role_embedding_v1", "dim": DIM, "one_hot_roles": []}, k_values=[K])
pipe = res["pipe"]
assert int(pipe.grammar.tuple_size) == K and len(res["mapped_artifacts"]) == len(texts), (int(pipe.grammar.tuple_size), len(res["mapped_artifacts"]), len(texts))
fid = pipe.tuple_scores(res["mapped_artifacts"])
print(f"fit: path {res['effective_path']}, k={K}, dim={DIM}, qubits={res['selected'].get('n_qubits')}, features {pipe._T.shape[0]}; {len(texts)} texts", flush=True)

stats = collections.defaultdict(lambda: collections.defaultdict(list))
for i, p in enumerate(pairs):
    e = edit_index(p)
    for name, before, after in (("FidelityAI", fid[2 * i], fid[2 * i + 1]), ("window counts", count_worst(p["before"]), count_worst(p["after"]))):
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
    print(f"{kind} ({len(s['FidelityAI detect'])} edits): detect — FidelityAI {m(s['FidelityAI detect']):.3f}, counts {m(s['window counts detect']):.3f}; "
          f"locate — FidelityAI {m(s['FidelityAI locate']):.3f}, counts {m(s['window counts locate']):.3f}, chance {m(s['chance locate']):.3f}")
