"""The tuple as ScribeSLM itself: a suggestion's path through the pipeline, as closed
values (examples/pipeline_paths.rs): original meaning category and confidence, how the
substitute's meaning relates (same synset, parent, same category, other, none), its
category and confidence in place, part of speech, relation, and optionally the judge's
gain. Rulebook: real Simple English Wikipedia swaps between WordNet synonyms, human
choices, one-class. Candidates: SWORDS (publishable judge) and ProLex (casual), scored
only. Within target (0.5 chance), all pairs and synonym pairs. A logistic model on pair
differences, held out by target, asks whether FidelityAI adds to the judge's gain.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_pipeline_paths.py <rulebook jsonl> <candidates jsonl> <wordnet jsonl> <engine dir>
"""
import collections, json, sys

import numpy as np

rule_path, cand_path, wn_path, engine = sys.argv[1:5]
sys.path.insert(0, engine)
from fidelityai.fitter import HandoffPipeline  # noqa: E402
from fidelityai.grammar import GrammarSpec  # noqa: E402
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
import types  # noqa: E402

wn, syn = {}, collections.defaultdict(set)
for l in open(wn_path):
    s = json.loads(l)
    wn[s["id"]] = s
    for m in s["members"]:
        syn[m.lower()].add(s["id"])
load = lambda p: [json.loads(l) for l in open(p)]
rule, cands = load(rule_path), load(cand_path)
rule = [r for r in rule if len(r.get("judge") or []) == 2]
cands = [r for r in cands if len(r.get("judge") or []) == 2]
gain = lambda r: r["judge"][1] - r["judge"][0]
qs = lambda xs: np.quantile(xs, [0.2, 0.4, 0.6, 0.8]).tolist()
GAIN, GAP = qs([gain(r) for r in rule]), qs([r["gap"] for r in rule if r.get("gap") is not None])
SUBGAP = qs([r["sub_gap"] for r in rule if r.get("sub_gap") is not None])
band = lambda v, e: "-" if v is None else str(sum(v > x for x in e))


def relation(r):
    a, b = r.get("meaning"), r.get("sub_meaning")
    if not a or not b:
        return "none"
    if a["s"] == b["s"]:
        return "same"
    sa, sb = wn.get(a["s"], {}), wn.get(b["s"], {})
    if b["s"] in sa.get("hypernyms", []) or a["s"] in sb.get("hypernyms", []):
        return "parent"
    if set(sa.get("hypernyms", [])) & set(sb.get("hypernyms", [])):
        return "sibling"
    return "category" if a["c"] == b["c"] else "other"


def path(r, with_judge):
    out = [("m", (r.get("meaning") or {}).get("c", "-")), ("g", band(r.get("gap"), GAP)), ("r", relation(r)), ("s", (r.get("sub_meaning") or {}).get("c", "-")),
           ("h", band(r.get("sub_gap"), SUBGAP)), ("p", r["pos"]), ("l", r["rel"])]
    return out + ([("j", band(gain(r), GAIN))] if with_judge else [])


groups = collections.defaultdict(lambda: ([], []))
for i, c in enumerate(cands):
    groups[c["tid"]][1 - c["y"]].append(i)
pairs = [(a, b, t) for t, (acc, rej) in groups.items() for a in acc for b in rej]
is_syn = lambda c: syn[c["substitute"].lower()] & syn[c["target"].lower()]
syn_pairs = [p for p in pairs if is_syn(cands[p[0]]) and is_syn(cands[p[1]])]
rate = lambda s, ps: sum((s[a] > s[b]) + 0.5 * (s[a] == s[b]) for a, b, _ in ps) / max(len(ps), 1)


def show(name, s):
    print(f"  {name}: all {rate(s, pairs):.3f}, synonyms {rate(s, syn_pairs):.3f}", flush=True)


def adds(name, s, base):
    rng = np.random.default_rng(3)
    sign = rng.choice([-1, 1], len(pairs))
    d = lambda v: np.array([v[a] - v[b] for a, b, _ in pairs]) * sign
    X = np.c_[d(base), d(s)]
    X = X / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
    y, t = (sign > 0).astype(int), np.array([p[2] for p in pairs])
    a, b = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, t):
        a[te] = LogisticRegression(fit_intercept=False).fit(X[tr][:, [0]], y[tr]).decision_function(X[te][:, [0]])
        b[te] = LogisticRegression(fit_intercept=False).fit(X[tr], y[tr]).decision_function(X[te])
    print(f"    {name}, adding to the judge: held-out AUC {roc_auc_score(y, a):.3f} -> {roc_auc_score(y, b):.3f}", flush=True)


print(f"rulebook {len(rule)} editor swaps; {len(cands)} candidates, {len(pairs)} pairs, {len(syn_pairs)} of synonyms", flush=True)
judge = np.array([gain(c) for c in cands])
print("  relation shares, rulebook:", dict(collections.Counter(relation(r) for r in rule).most_common()))
show("judge slot gain", judge)
RANK = {"same": 4, "parent": 3, "sibling": 2, "category": 1, "other": 0, "none": 0}
show("rule: meaning relation", np.array([RANK[relation(c)] for c in cands], dtype=float))
for wj in (False, True):
    tag = "with judge" if wj else "no judge"
    key = lambda r: tuple(path(r, wj))
    counts = collections.Counter(key(r) for r in rule)
    show(f"counts of the whole path, {tag}", np.array([counts[key(c)] for c in cands], dtype=float))
    # Automatic choice, as customers get it.
    rec = lambda r: {f: v for f, v in path(r, wj)}
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[rec(r) for r in rule], label="paths"),
                             artifact_dataset=types.SimpleNamespace(episodes=[rec(c) for c in cands], label="c"), max_qubits=20, seed=42)
    s = np.array([-x for x in res["artifact_scores"]])
    show(f"FidelityAI auto, {tag} (k={res['selected'].get('k')})", s)
    if wj: adds("auto", s, judge)
    for K, DIM in ((4, 8), (len(path(rule[0], wj)), 4)):
        ids = {}
        enc = lambda r: [ids.setdefault(t, len(ids)) for t in path(r, wj)][: len(path(r, wj)) // K * K]
        src, test = [enc(r) for r in rule], [enc(c) for c in cands]
        g = GrammarSpec(roles=[f"t{i}" for i in range(len(ids))], allowed_transitions={i: [] for i in range(len(ids))}, forbidden_tuples=set(), benign_tuples=set(),
                        tuple_size=K, n_qubits=K * 4, label="paths", encoding={"type": "role_embedding_v1", "dim": DIM, "one_hot_roles": []})
        pipe = HandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42).fit(src)
        s = -np.array(pipe.score(test))
        show(f"FidelityAI k={K} dim={DIM}, {tag} (projection {pipe._T.shape[1]})", s)
        if wj: adds(f"k={K}", s, judge)
