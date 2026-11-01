"""Does a piece agree with itself? Closed writing choices per sentence
(examples/consistency_features.rs); the rulebook is five consecutive real OANC sentences at a time. A control
built from another human-written corpus: a real held-out passage against the same
passage with one middle sentence replaced by a real sentence from Simple English
Wikipedia or Tatoeba, of similar length, whose tuple is common in the rulebook, so
it passes modularly and only the mix is off. Local and free.

Scored: the original is more typical than its spliced version (0.5 chance), by
FidelityAI's document record, worst tuple and inter-tuple tier, against k-gram counts
(whole tuples, worst and mean), adjacent tuple-pair counts, and how many distinct values
the passage mixes. Writes pairs for the judge.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_consistency_eval.py <core> <engine> <work> <factors,...> <sizes> [pairs = 500]
"""
import collections, json, math, random, re, subprocess, sys

core, engine, work, factor_sets, sizes = sys.argv[1:6]
n_pairs = int(sys.argv[6]) if len(sys.argv) > 6 else 500
sys.path.insert(0, engine)
from fidelityai.fitter import HandoffPipeline, HierarchicalHandoffPipeline  # noqa: E402
from fidelityai.grammar import GrammarSpec  # noqa: E402

split = lambda p: [x.strip() for x in re.split(r"(?<=[.!?])\s+", p) if x.strip()]


def parse(texts, name):
    with open(f"{work}/{name}.txt", "w") as f:
        f.write("\n".join(texts) + "\n")
    subprocess.run([f"{core}/target/release/examples/consistency_features", f"{work}/{name}.txt", f"{work}/{name}.jsonl"], cwd=core, check=True)
    return [json.loads(l) for l in open(f"{work}/{name}.jsonl")]


passages, run = [], []
for line in open(f"{core}/data/corpus/oanc_written.txt"):
    for x in split(line.strip()):
        if len(x.split()) > 45:
            run = []
            continue
        run.append(x)
        if len(run) == 5:
            passages.append(" ".join(run)); run = []
passages = list(dict.fromkeys(passages))
random.Random(1).shuffle(passages)
held, rule_text = passages[:n_pairs * 3], passages[n_pairs * 3:n_pairs * 3 + max(map(int, sizes.split(",")))]
print(f"{len(passages)} passages; rulebook pool {len(rule_text)}", flush=True)
others = [l.strip() for l in open(f"{core}/data/corpus/tatoeba.txt") if 6 <= len(l.split()) <= 30]
r0 = random.Random(2)
others = r0.sample(others, 30000) + [l.rstrip("\n").split("\t")[3] for l in open(f"{core}/data/simplewiki-history/stable_test.tsv").readlines()[:15000]]
other_f = parse(others, "others")
rule_all = parse(rule_text, "rule")

for factors in factor_sets.split(","):
    K = len(factors)
    DIM = {2: 16, 3: 16, 4: 8}[K]
    tup = lambda c: tuple(f"{f}:{c[f]}" for f in factors)
    full = collections.Counter(tup(c) for d in rule_all for c in d)
    common = sorted(full.values())[len(full) // 2] if full else 0
    pool = collections.defaultdict(list)
    for s, d in zip(others, other_f):
        if len(d) == 1 and full[tup(d[0])] >= common:
            pool[len(s.split()) // 4].append(s)
    r = random.Random(3)
    origs, spliced = [], []
    for p in held:
        x = split(p)
        if len(x) < 3:
            continue
        i = r.randrange(1, len(x) - 1)
        b = pool.get(len(x[i].split()) // 4)
        if not b:
            continue
        origs.append(p); spliced.append(" ".join(x[:i] + [r.choice(b)] + x[i + 1:]))
        if len(origs) == n_pairs:
            break
    with open(f"{work}/judge_{factors}.tsv", "w") as f:
        for n, (a, b) in enumerate(zip(origs, spliced)):
            f.write(f"{n}\t1\t{a}\n{n}\t0\t{b}\n")
    O, X = parse(origs, "o" + factors), parse(spliced, "x" + factors)
    keep = [i for i in range(len(O)) if O[i] and X[i]]
    print(f"\nfactors {factors}: {len(keep)} pairs; middle-frequency bar {common}", flush=True)
    for n_rule in map(int, sizes.split(",")):
        rule = [d for d in rule_all[:n_rule] if d]
        ids = {}
        enc = lambda t: [ids.setdefault(v, len(ids)) for v in t]
        groups = [[enc(tup(c)) for c in d] for d in rule]
        docs = [[enc(tup(c)) for c in d] for i in keep for d in (O[i], X[i])]
        g = GrammarSpec(roles=[f"t{i}" for i in range(len(ids))], allowed_transitions={i: [] for i in range(len(ids))}, forbidden_tuples=set(), benign_tuples=set(),
                        tuple_size=K, n_qubits=K * 4 + 2, label="choices", encoding={"type": "role_embedding_v1", "dim": DIM, "one_hot_roles": []})
        flat = lambda d: [v for t in d for v in t]
        pipe = HandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42).fit([flat(d) for d in groups])
        record = -pipe.score([flat(d) for d in docs])
        worst = [-max(t) if t else None for t in pipe.tuple_scores([flat(d) for d in docs])]
        # Inter-tuple tier: sentence tuples grouped by passage.
        sp = HandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42).fit([t for d in groups for t in d])
        hier = HierarchicalHandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42, intra_pipe=sp).fit(groups)
        inter = -hier.score(docs)
        direct = -hier.direct_z_score(docs)
        counts = collections.Counter(tup(c) for d in rule_all[:n_rule] for c in d)
        adj = collections.Counter((tup(a), tup(b)) for d in rule_all[:n_rule] for a, b in zip(d, d[1:]))
        total = max(sum(counts.values()), 1)
        kv = [(i, side) for i in keep for side in (O[i], X[i])]
        mean_k = [sum(math.log1p(counts[tup(c)]) for c in d) / len(d) for _, d in kv]
        worst_k = [min(counts[tup(c)] for c in d) for _, d in kv]
        pair_k = [sum(math.log1p(adj[(tup(a), tup(b))]) for a, b in zip(d, d[1:])) / max(len(d) - 1, 1) for _, d in kv]
        mix = [-sum(len({c[f] for c in d}) for f in factors) for _, d in kv]

        def rate(v):
            return sum((v[2 * m] > v[2 * m + 1]) + 0.5 * (v[2 * m] == v[2 * m + 1]) for m in range(len(keep)) if v[2 * m] is not None and v[2 * m + 1] is not None) / max(len(keep), 1)

        print(f"  rulebook {len(rule)} (projection {pipe._T.shape[1]}): " + ", ".join(f"{k} {rate(v):.3f}" for k, v in (
            ("record", record), ("worst tuple", worst), ("inter-tuple", inter), ("inter direct", direct), ("k-gram mean", mean_k), ("k-gram worst", worst_k), ("adjacent pairs", pair_k), ("mix", mix))), flush=True)
