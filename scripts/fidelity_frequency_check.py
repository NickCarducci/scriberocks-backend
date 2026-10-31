"""Does FidelityAI's inter-tuple tier add anything beyond word frequency? SWORDS test
and dev combined; a sentence is its parser-linked word tuples, grouped by sentence;
rulebook of OANC sentences. Within target (0.5 chance): all pairs and synonym pairs
(fidelity_misnomer_eval.py), overall and with the two substitutes' corpus frequencies
within a factor of 2 and 4. Also a cross-validated logistic model on pair differences,
held out by target: frequency alone, against frequency plus FidelityAI. Writes the
similar-frequency pairs for the judge.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_frequency_check.py <core> <engine> <work> <rulebook sentences> <subs.tsv>:<synonym pairs tsv> ...
"""
import collections, json, math, random, re, subprocess, sys

import numpy as np

core, engine, work, n_rule = sys.argv[1:5]
n_rule = int(n_rule)
sys.path.insert(0, engine)
from fidelityai.fitter import HandoffPipeline, HierarchicalHandoffPipeline  # noqa: E402
from fidelityai.grammar import GrammarSpec  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

K, DIM = 3, 16


def parse(texts, name):
    with open(f"{work}/{name}.txt", "w") as f:
        f.write("\n".join(texts) + "\n")
    subprocess.run([f"{core}/target/release/examples/sentence_tuples", f"{work}/{name}.txt", f"{work}/{name}.jsonl"], cwd=core, check=True)
    return [json.loads(l) for l in open(f"{work}/{name}.jsonl")]


items, syn_pairs = [], set()
for spec in sys.argv[5:]:
    tsv, syn = spec.split(":")
    for l in open(tsv):
        r = l.rstrip("\n").split("\t")
        items.append({"tid": r[1], "y": int(r[2]), "sub": r[5].lower(), "text": " ".join(f"{r[3]} {r[5]} {r[6]}".split())})
    groups = collections.defaultdict(dict)
    for l in open(syn):
        n, y, t = l.rstrip("\n").split("\t")
        groups[n][int(y)] = t
    syn_pairs |= {(g[1], g[0]) for g in groups.values() if 0 in g and 1 in g}
corpus, freq = [], collections.Counter()
for line in open(f"{core}/data/corpus/oanc_written.txt"):
    freq.update(re.findall(r"[a-z']+", line.lower()))
    for s in re.split(r"(?<=[.!?])\s+", line.strip()):
        if 6 <= len(s.split()) <= 40:
            corpus.append(s)
random.Random(17).shuffle(corpus)
rule = parse(corpus[:n_rule], "rule")
cands = parse([x["text"] for x in items], "cands")
ids = {}
tup = lambda t: [ids.setdefault((p, v), len(ids)) for p, v in enumerate(t)]
source = [[tup(t) for t in d] for d in rule if d]
test = {i: [tup(t) for t in d] for i, d in enumerate(cands) if d}
g = GrammarSpec(roles=[f"t{i}" for i in range(len(ids))], allowed_transitions={i: [] for i in range(len(ids))}, forbidden_tuples=set(), benign_tuples=set(),
                tuple_size=K, n_qubits=K * 4 + 2, label="tuples", encoding={"type": "role_embedding_v1", "dim": DIM, "one_hot_roles": []})
pipe = HandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42).fit([t for d in source for t in d])
hier = HierarchicalHandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42, intra_pipe=pipe).fit(source)
docs = sorted(test)
score = {d: -float(x) for d, x in zip(docs, hier.score([test[d] for d in docs]))}
logf = {i: math.log1p(freq[x["sub"]]) for i, x in enumerate(items)}
groups = collections.defaultdict(lambda: ([], []))
for i, x in enumerate(items):
    if i in score:
        groups[x["tid"]][1 - x["y"]].append(i)
pairs = [(a, b, t) for t, (acc, rej) in groups.items() for a in acc for b in rej]
is_syn = lambda p: (items[p[0]]["text"], items[p[1]]["text"]) in syn_pairs
ratio = lambda p: abs(logf[p[0]] - logf[p[1]])


def rate(s, ps):
    return sum((s[a] > s[b]) + 0.5 * (s[a] == s[b]) for a, b, _ in ps) / max(len(ps), 1)


def binomial_se(ps):
    return (0.25 / max(len(ps), 1)) ** 0.5


print(f"rulebook {len(source)} sentences; {len(pairs)} pairs, {sum(map(is_syn, pairs))} of synonyms", flush=True)
for name, base in (("all", pairs), ("synonyms", [p for p in pairs if is_syn(p)])):
    for label, ps in (("any frequency", base), ("within 4x", [p for p in base if ratio(p) <= math.log(4)]), ("within 2x", [p for p in base if ratio(p) <= math.log(2)])):
        print(f"{name}, {label} ({len(ps)}; se about {binomial_se(ps):.3f}): FidelityAI {rate(score, ps):.3f}, word frequency {rate(logf, ps):.3f}", flush=True)
        slug = f"{name}_{label.replace(' ', '')}"
        with open(f"{work}/judge_{slug}.tsv", "w") as f:
            for n, (a, b, _) in enumerate(ps):
                f.write(f"{n}\t1\t{items[a]['text']}\n{n}\t0\t{items[b]['text']}\n")


def cross_val(ps, label):
    rng = random.Random(3)
    rows = []
    for a, b, t in ps:
        y, sign = (1, 1) if rng.random() < 0.5 else (0, -1)
        rows.append((sign * (logf[a] - logf[b]), sign * (score[a] - score[b]), y, t))
    X = np.array([[r[0], r[1]] for r in rows])
    X = X / X.std(axis=0)
    y, t = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    out = {"frequency": [0, 0], "frequency + FidelityAI": [0, 0]}
    preds = {k: np.zeros(len(y)) for k in out}
    for tr, te in GroupKFold(5).split(X, y, t):
        for k, cols in (("frequency", [0]), ("frequency + FidelityAI", [0, 1])):
            m = LogisticRegression(fit_intercept=False).fit(X[tr][:, cols], y[tr])
            preds[k][te] = m.decision_function(X[te][:, cols])
    m = LogisticRegression(fit_intercept=False).fit(X, y)
    print(f"model, {label} ({len(y)}): held-out AUC frequency {roc_auc_score(y, preds['frequency']):.3f}, frequency + FidelityAI {roc_auc_score(y, preds['frequency + FidelityAI']):.3f}; "
          f"weights frequency {m.coef_[0][0]:.3f}, FidelityAI {m.coef_[0][1]:.3f}", flush=True)


cross_val(pairs, "all")
syn = [p for p in pairs if is_syn(p)]
if len(syn) >= 50:
    cross_val(syn, "synonyms")
