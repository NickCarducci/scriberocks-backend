"""Meaning learned upstream: FidelityAI's word positions use the vectors casual v2
learned from standard English, reduced to fit, instead of co-occurrence in the
rulebook or hand-written descriptions, so a pairing the rulebook never saw still
has meaningful overlaps. Records: real (governor, relation, word) pairings
(examples/collocations.rs); candidates: substitutes judged by people. Fit on real
text only; ProLex stays local. Within target: an accepted substitute beats a
rejected one (0.5 chance), over all pairs, pairs where both substitutes' pairing is
absent from the rulebook, and pairs of similar word frequency (within a factor of 2).

usage: [VECTORS=<words in context json>] ~/.venvs/fidelityai/bin/python -I fidelity_usage_vectors_eval.py <colloc prefix> <model dir> <engine dir> [rulebook = 5000] [refs = 30]
"""
import collections, json, math, os, random, struct, sys

import numpy as np

prefix, model, engine = sys.argv[1:4]
n_rule = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
refs = int(sys.argv[5]) if len(sys.argv) > 5 else 30
DIM, K = 16, 3
sys.path.insert(0, engine)
from fidelityai import role_embedding as re_mod  # noqa: E402
from fidelityai.fitter import HandoffPipeline  # noqa: E402
from fidelityai.grammar import GrammarSpec  # noqa: E402


def piece_table():
    with open(f"{model}/model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        key = next(k for k in header if "word_embeddings" in k)
        meta = header[key]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        data = np.frombuffer(f.read(end - start), dtype={"F32": np.float32, "F16": np.float16}[meta["dtype"]]).reshape(meta["shape"])
    return np.asarray(data, dtype=float)


table = piece_table()
vocab = {w: i for i, w in enumerate(open(f"{model}/vocab.txt").read().split("\n"))}


def pieces(word):
    out, i = [], 0
    while i < len(word):
        for j in range(len(word), i, -1):
            p = ("##" if i else "") + word[i:j]
            if p in vocab:
                out.append(vocab[p]); i = j; break
        else:
            return []
    return out


context = json.load(open(os.environ["VECTORS"])) if os.environ.get("VECTORS") else None  # examples/context_vectors.rs


def word_vector(word):
    if context is not None:
        v = context.get(word.lower())
        return np.asarray(v, dtype=float) if v else None
    p = pieces(word.lower())
    return table[p].mean(axis=0) if p else None


cands = [json.loads(l) for l in open(f"{prefix}.candidates.jsonl")]
labels = collections.defaultdict(set)
for c in cands:
    labels[c["tid"]].add(c["y"])
cands = [c for c in cands if labels[c["tid"]] == {0, 1} and c["up"]]
triple = lambda t: (t["gov"], t["rel"], t["dep"])
up_keys = {(c["up"]["gov"], c["up"]["rel"]) for c in cands}
by_key = collections.defaultdict(list)
for line in open(f"{prefix}.rulebook.tsv"):
    g, r, d, n = line.rstrip("\n").split("\t")
    if (g, r) in up_keys:
        by_key[(g, r)].append((d, int(n)))
rng = random.Random(13)
keys = sorted(by_key)
per = max(1, n_rule // max(len(keys), 1))
rule = []
for k in keys:
    ds, ws = zip(*by_key[k])
    rule += [(k[0], k[1], d) for d in rng.choices(ds, weights=ws, k=per)]
rng.shuffle(rule)
rule = rule[:n_rule]
counts = collections.Counter(rule)
ids = {}
tid = lambda pos, v: ids.setdefault((pos, v), len(ids))
seq = lambda t: [tid(0, t[0]), tid(1, t[1]), tid(2, t[2])]
source = [seq(t) for t in rule]
test = [seq(triple(c["up"])) for c in cands]
inverse = {i: k for k, i in ids.items()}

# Reduce the model's word vectors to DIM - 1 numbers (the last stays for unseen words).
words = sorted({v for (p, v) in ids if p != 1})
have = {w: v for w in words if (v := word_vector(w)) is not None}
matrix = np.stack(list(have.values()))
mean = matrix.mean(axis=0)
_, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
basis = vt[: DIM - 1]
reduced = {w: (v - mean) @ basis.T for w, v in have.items()}
print(f"{len(cands)} candidates for {len({c['tid'] for c in cands})} targets; rulebook {len(rule)} pairings; "
      f"{len(have)}/{len(words)} words have vectors", flush=True)


def pipeline(encoding, n_qubits=K * 4 + 2):
    g = GrammarSpec(roles=[f"t{i}" for i in range(len(ids))], allowed_transitions={i: [] for i in range(len(ids))}, forbidden_tuples=set(), benign_tuples=set(),
                    tuple_size=K, n_qubits=n_qubits, label="pairings", encoding=encoding)
    return HandoffPipeline(g, max_benign_refs=refs, classical_nu=0.1, seed=42).fit(source)


learned = pipeline({"type": "role_embedding_v1", "dim": DIM, "one_hot_roles": []})
lt = learned.grammar.encoding
tables = [{}, lt["tables"][1], {}]
for i, (p, v) in inverse.items():
    if p != 1 and v in reduced:
        u = reduced[v]
        n = np.linalg.norm(u)
        if n:
            tables[p][i] = (u / n).tolist() + [0.0]
usage = pipeline({"type": "role_embedding_v1", "k": K, "dims": [DIM] * K, "tables": tables, "unknown": [[0.0] * (DIM - 1) + [1.0]] * K, "dim": DIM, "one_hot_roles": []})
scores = {"usage vectors": -np.array(usage.score(test)), "co-occurrence": -np.array(learned.score(test)),
          "counts": np.array([counts[triple(c["up"])] for c in cands], dtype=float),
          "word frequency": np.array([math.log1p(c["up"]["dep_n"]) for c in cands])}
print(f"projection: usage {usage._T.shape[1]} dims, co-occurrence {learned._T.shape[1]} dims", flush=True)


def within(s, keep):
    groups = collections.defaultdict(lambda: ([], []))
    for i, c in enumerate(cands):
        groups[c["tid"]][1 - c["y"]].append(i)
    wins = n = 0
    for acc, rej in groups.values():
        for a in acc:
            for b in rej:
                if keep(a, b):
                    wins += (s[a] > s[b]) + 0.5 * (s[a] == s[b]); n += 1
    return wins / max(n, 1), n


unseen = lambda i: counts[triple(cands[i]["up"])] == 0
freq = [cands[i]["up"]["dep_n"] for i in range(len(cands))]
close = lambda a, b: max(freq[a], freq[b]) <= 2 * max(min(freq[a], freq[b]), 1)
for label, keep in (("all pairs", lambda a, b: True), ("both unseen", lambda a, b: unseen(a) and unseen(b)), ("similar frequency", close)):
    n = within(scores["counts"], keep)[1]
    print(f"{label} ({n}): " + ", ".join(f"{m} {within(s, keep)[0]:.3f}" for m, s in scores.items()), flush=True)
