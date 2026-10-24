"""Local, free FidelityAI fit on the edit records (fidelity_edit_records.py),
the same cascade quantum.fidelityai.org runs (abcd_fit_and_score), plus a
classical count baseline. Reports AUC: how often a real held-out edit is
scored more typical than a random one (0.5 chance, 1.0 perfect).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_local_fit.py <records dir> <FidelityAI engine dir> [max qubits = 20]
"""
import collections, json, math, sys, time, types

rec_dir, engine = sys.argv[1], sys.argv[2]
max_qubits = int(sys.argv[3]) if len(sys.argv) > 3 else 20
sys.path.insert(0, engine)
load = lambda name: [json.loads(l) for l in open(f"{rec_dir}/{name}.jsonl")]
source, real, rand = load("source"), load("test_real"), load("test_random")
candidates = real + rand
labels = [1] * len(real) + [0] * len(rand)  # 1 = real editor edit


def auc(typicality):
    """P(a real edit is more typical than a random one); ties count half."""
    pos = [t for t, y in zip(typicality, labels) if y]
    neg = [t for t, y in zip(typicality, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# Classical baseline: how often editors make this change, backing off to the
# inserted and removed words on their own (add-one smoothing, log scale).
full = collections.Counter((r["op"], r["removed"], r["inserted"]) for r in source)
ins = collections.Counter(r["inserted"] for r in source)
rem = collections.Counter(r["removed"] for r in source)
ctx = collections.Counter((r["left"], r["inserted"], r["right"]) for r in source)
n = len(source)
count_score = [
    math.log(1 + full[(r["op"], r["removed"], r["inserted"])]) * 3
    + math.log(1 + ctx[(r["left"], r["inserted"], r["right"])])
    + math.log((1 + ins[r["inserted"]]) / n) + math.log((1 + rem[r["removed"]]) / n)
    for r in candidates
]
print(f"count baseline: AUC {auc(count_score):.3f} (n = {len(real)} real, {len(rand)} random)")

from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

t = time.time()
# Source: one episode of rows, as the API wraps an uploaded table; capped at
# the engine's recommended 5,000 records. Candidates: one episode each, so
# each gets its own score.
fit_source = source[:5000]
result = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[fit_source], label="edits"),
                            artifact_dataset=types.SimpleNamespace(episodes=[[r] for r in candidates], label="candidates"),
                            max_qubits=max_qubits, seed=42)
scores = result["artifact_scores"]
sel = result["selected"]
print(f"FidelityAI (local): path {result['effective_path']}, k={sel.get('k')}, v={sel.get('v')}, qubits={sel.get('n_qubits')}, "
      f"{len(scores)} scores for {len(candidates)} candidates, {time.time() - t:.0f} s")
# Each score maps back to its candidate row (provenance sourceIndex); a row
# may yield several frames (mean) or none (left out, and counted).
per = collections.defaultdict(list)
for sc, pv in zip(scores, result.get("artifact_provenance") or []):
    if sc is not None and isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int):
        per[pv["sourceIndex"]].append(sc)
covered = [i for i in range(len(candidates)) if per.get(i)]
print(f"FidelityAI (local): {len(covered)} of {len(candidates)} candidates scored")
if covered:
    labels_all = labels
    labels = [labels_all[i] for i in covered]
    # Scores are distances from the fit: larger = less typical.
    print(f"FidelityAI (local): AUC {auc([-sum(per[i]) / len(per[i]) for i in covered]):.3f} on the scored candidates")
    print(f"count baseline on the same candidates: AUC {auc([count_score[i] for i in covered]):.3f}")

# Second encoding: each edit as integer tokens (op, removed, inserted, left,
# right; words seen fewer than 3 times share one id) through path D, with
# role embeddings (capacity independent of vocabulary) and the op one-hot.
cols = ["op", "removed", "inserted", "left", "right"]
seen = collections.Counter(r[c] for r in source[:5000] for c in cols)
ids = {"<rare>": 0}
for w, c in seen.most_common():
    if c >= 3: ids.setdefault(w, len(ids))
enc = lambda r: [ids.get(r[c], 0) for c in cols]
t = time.time()
res2 = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[[{"sequence": enc(r)}] for r in source[:5000]], label="edits-ids"),
                          artifact_dataset=types.SimpleNamespace(episodes=[[{"sequence": enc(r)}] for r in candidates], label="candidates-ids"),
                          max_qubits=max_qubits, seed=42, encoding={"type": "role_embedding_v1", "dim": 16, "one_hot_roles": [0]})
s2 = res2["artifact_scores"]
print(f"FidelityAI role embeddings: path {res2['effective_path']}, k={res2['selected'].get('k')}, qubits={res2['selected'].get('n_qubits')}, "
      f"{len(s2)} scores for {len(candidates)} candidates, vocabulary {len(ids)}, {time.time() - t:.0f} s")
if len(s2) == len(candidates):
    labels = labels_all
    print(f"FidelityAI role embeddings: AUC {auc([-x for x in s2]):.3f}")
