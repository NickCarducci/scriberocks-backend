"""Word tuples in sequence: a sentence as all its parser-linked (governor, relation,
word) tuples, not just those touching one word. Fit on real OANC sentences; candidates
are SWORDS sentences with a human-judged substitute in place. Scored by FidelityAI's
sentence record, its worst tuple and, with tuples grouped by sentence, its inter-tuple
tier. Within target (0.5 chance), on all pairs and on pairs where both substitutes are
WordNet synonyms of the original word (texts from fidelity_misnomer_eval.py).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_sequence_eval.py <core> <subs.tsv> <synonym pairs tsv> <engine> <work> <sizes> <grouped size>
"""
import collections, json, random, re, subprocess, sys, types

core, tsv, syn_tsv, engine, work, sizes, grouped = sys.argv[1:8]
sys.path.insert(0, engine)
from fidelityai.fitter import HandoffPipeline, HierarchicalHandoffPipeline  # noqa: E402
from fidelityai.grammar import GrammarSpec  # noqa: E402
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

K, DIM = 3, 16
ENC = {"type": "role_embedding_v1", "dim": DIM, "one_hot_roles": []}


def parse(texts, name):
    with open(f"{work}/{name}.txt", "w") as f:
        f.write("\n".join(texts) + "\n")
    subprocess.run([f"{core}/target/release/examples/sentence_tuples", f"{work}/{name}.txt", f"{work}/{name}.jsonl"], cwd=core, check=True)
    return [json.loads(l) for l in open(f"{work}/{name}.jsonl")]


rows = [l.rstrip("\n").split("\t") for l in open(tsv)]
items = [(r[1], int(r[2]), " ".join(f"{r[3]} {r[5]} {r[6]}".split())) for r in rows]
cands = parse([t for _, _, t in items], "cands_" + tsv.split("/")[-1])
syn_pairs = collections.defaultdict(dict)
for l in open(syn_tsv):
    n, y, t = l.rstrip("\n").split("\t")
    syn_pairs[n][int(y)] = t
syn_pairs = [(p[1], p[0]) for p in syn_pairs.values() if 0 in p and 1 in p]
corpus = []
for line in open(f"{core}/data/corpus/oanc_written.txt"):
    for s in re.split(r"(?<=[.!?])\s+", line.strip()):
        if 6 <= len(s.split()) <= 40:
            corpus.append(s)
random.Random(17).shuffle(corpus)
all_rule = parse(corpus[:max(map(int, sizes.split(",")))], "rule")


def report(label, typical):
    """typical: index -> score (higher = more typical)."""
    groups = collections.defaultdict(lambda: ([], []))
    for i, (t, y, _) in enumerate(items):
        if typical.get(i) is not None:
            groups[t][1 - y].append(typical[i])
    wins = n = 0
    for acc, rej in groups.values():
        for a in acc:
            for b in rej:
                wins += (a > b) + 0.5 * (a == b); n += 1
    by_text = {items[i][2]: v for i, v in typical.items()}
    sw = [(by_text[a] > by_text[b]) + 0.5 * (by_text[a] == by_text[b]) for a, b in syn_pairs if a in by_text and b in by_text]
    print(f"  {label}: all {wins / max(n, 1):.3f} ({n}); synonyms {sum(sw) / max(len(sw), 1):.3f} ({len(sw)})", flush=True)


for n_rule in map(int, sizes.split(",")):
    ids = {}
    enc = lambda doc: [ids.setdefault((p, v), len(ids)) for t in doc for p, v in enumerate(t)]
    rule = [s for s in map(enc, all_rule[:n_rule]) if s]
    test = [enc(d) for d in cands]
    keep = [i for i, s in enumerate(test) if s]
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[[{"sequence": s}] for s in rule], label="sentences"),
                             artifact_dataset=types.SimpleNamespace(episodes=[[{"sequence": test[i]}] for i in keep], label="candidates"),
                             max_qubits=20, seed=42, encoding=ENC, k_values=[K])
    print(f"sentence records, rulebook {len(rule)}, projection {res['pipe']._T.shape[1]}, v={len(ids)}", flush=True)
    assert len(res["mapped_artifacts"]) == len(keep)
    report("record", {i: -s for i, s in zip(keep, res["artifact_scores"]) if s is not None})
    report("worst tuple", {i: -max(t) for i, t in zip(keep, res["pipe"].tuple_scores(res["mapped_artifacts"])) if t})

n_rule = int(grouped)
ids = {}
tup = lambda t: [ids.setdefault((p, v), len(ids)) for p, v in enumerate(t)]
source = [[tup(t) for t in doc] for doc in all_rule[:n_rule] if doc]
test = {i: [tup(t) for t in doc] for i, doc in enumerate(cands) if doc}
g = GrammarSpec(roles=[f"t{i}" for i in range(len(ids))], allowed_transitions={i: [] for i in range(len(ids))}, forbidden_tuples=set(), benign_tuples=set(),
                tuple_size=K, n_qubits=K * 4 + 2, label="tuples", encoding=dict(ENC))
pipe = HandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42).fit([t for doc in source for t in doc])
hier = HierarchicalHandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42, intra_pipe=pipe).fit(source)
docs = sorted(test)
print(f"tuples grouped by sentence, rulebook {len(source)} sentences", flush=True)
report("inter-tuple tier", {d: -float(x) for d, x in zip(docs, hier.score([test[d] for d in docs]))})
report("inter-tuple direct", {d: -float(x) for d, x in zip(docs, hier.direct_z_score([test[d] for d in docs]))})
