"""Fit the sense-stage records locally with FidelityAI's engine
(role_embedding_v1) and score held-out cases.

  PYTHONPATH=<FidelityAI/engine> python scripts/fidelity_sense_eval.py data/fidelity/senses [dim] [max-tests]
"""
import json, random, sys, time
from collections import Counter
import numpy as np
from fidelityai.fitter import HandoffPipeline
from fidelityai.grammar import GrammarSpec

d = sys.argv[1]
dim = int(sys.argv[2]) if len(sys.argv) > 2 else 16
max_tests = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
vocab = json.load(open(f"{d}/vocab.json"))
lemma_rank = vocab.get("layout") == "lemma_rank"
sense_id = vocab["senses"]
records = [json.loads(l)["sequence"] for l in open(f"{d}/source.jsonl")]
k = len(records[0])
train_count = Counter(tuple(r[1:k - 1]) for r in records)
v = max(max(r) for r in records) + 1
g = GrammarSpec(roles=[f"token_{i}" for i in range(v)], allowed_transitions={}, forbidden_tuples=set(), benign_tuples=set(),
                tuple_size=k, n_qubits=4, window_arity=k,
                encoding={"type": "role_embedding_v1", "dim": dim, "one_hot_roles": [2] if lemma_rank else []})
t = time.perf_counter()
refs = int(sys.argv[4]) if len(sys.argv) > 4 else 30
pipe = HandoffPipeline(g, seed=42, max_benign_refs=refs).fit(records)
print(f"refs {refs}: fit {len(records)} records in {time.perf_counter() - t:.1f}s; {g.n_qubits} qubits; feature dim {pipe._T.shape[0]}")

tests = [json.loads(l) for l in open(f"{d}/tests.jsonl")]
usable = []
for c in tests:
    modeled = [s for s in c["candidates"] if s in sense_id]
    if len(modeled) >= 2 and any(gs in modeled for gs in c["gold"]):
        usable.append((c, modeled))
print(f"{len(usable)} of {len(tests)} held-out cases have the true sense and a rival in training")
random.Random(3).shuffle(usable)
usable = usable[:max_tests]
seqs, spans = [], []
for c, modeled in usable:
    spans.append((len(seqs), len(modeled)))
    if lemma_rank:
        seqs += [c["records"][c["candidates"].index(s)] for s in modeled]
    else:
        seqs += [[c["context"][0], sense_id[s], c["context"][1]] for s in modeled]
t = time.perf_counter()
scores = pipe.score(seqs)
print(f"scored {len(seqs)} candidate records in {time.perf_counter() - t:.1f}s")
right = first = major = 0
for (c, modeled), (start, n) in zip(usable, spans):
    pick = modeled[int(np.argmin(scores[start:start + n]))]
    right += pick in c["gold"]
    first += modeled[0] in c["gold"]          # WordNet order among modeled senses
    key = (lambda s: tuple(c["records"][c["candidates"].index(s)][1:k - 1])) if lemma_rank else (lambda s: (sense_id[s],))
    major += max(modeled, key=lambda s: train_count[key(s)]) in c["gold"]
n = len(usable)
print(f"{'lemma_rank' if lemma_rank else 'sense'} layout, dim {dim}: FidelityAI {100*right/n:.1f}%  first-listed {100*first/n:.1f}%  most-frequent-in-training {100*major/n:.1f}%  (n={n})")
