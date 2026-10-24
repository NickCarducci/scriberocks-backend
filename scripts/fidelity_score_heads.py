"""The earlier within-target tests, rescored with alternative FidelityAI score
heads from the same fitted pipeline (local and free):
  current   the deployed fit distance: |one-class score − source median|
  one-sided the one-class score above the source median (more typical → 0)
  residual  the part of the record outside the reference bank's span
            (reconstruction residual), relative to the source's
  combined  residual and one-sided, each scaled by its source 95th percentile

Tasks (within target; acceptable / accepted beats unacceptable / rejected):
  swords  rulebook SWORDS dev accepted, candidates SWORDS test
  prolex  rulebook SWORDS accepted (dev and test), candidates ProLex
  usage   rulebook casual v2's proposals for ProLex slots, candidates ProLex
          (local only: ProLex sentences are ETS-licensed)

usage: ~/.venvs/fidelityai/bin/python -I fidelity_score_heads.py <core data dir> <engine dir> [variants]
"""
import collections, json, random, sys, types

import numpy as np

data, engine = sys.argv[1], sys.argv[2]
variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["words", "synset", "relation"]
sys.path.insert(0, engine)
from fidelityai.fitter import extract_batch  # noqa: E402
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

load = lambda p: [json.loads(l) for l in open(f"{data}/fidelity-edits/{p}")]


def m(r, key, level):
    x = r.get(key)
    return x[level] if x else None


def record(r, variant):
    out = {"pos": r["pos"], "rel": r["rel"]}
    if variant == "words":
        out.update(target=r["target"], substitute=r["substitute"], governor=r["governor"])
    elif variant == "relation":
        a, b = r.get("meaning"), r.get("sub_meaning")
        out["relation"] = "unknown" if not (a and b) else "synset" if a["s"] == b["s"] else "category" if a["c"] == b["c"] else "other"
        out.update(meaning=m(r, "meaning", "c"), gov_meaning=m(r, "gov_meaning", "c"))
    else:
        out.update(target=r["target"], meaning=m(r, "meaning", "s"), sub_meaning=m(r, "sub_meaning", "s"), gov_meaning=m(r, "gov_meaning", "s"))
    return {k: v for k, v in out.items() if v is not None}


def heads(source, cands, variant):
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[record(r, variant) for r in source], label="s"),
                             artifact_dataset=types.SimpleNamespace(episodes=[record(r, variant) for r in cands], label="c"), max_qubits=20, seed=42)
    pipe = res["pipe"]

    def raw(seqs):
        feats, _ = extract_batch(seqs, pipe.grammar, pipe._active_dictionary_profile(), fit_scaler=False, scaler=pipe._scaler)
        z = feats @ pipe._T
        return pipe._ocsvm_boundary_score(z), pipe._reconstruction_residual(feats, z)

    b_src, r_src = raw(res["mapped_source"])
    b, r = raw(res["mapped_artifacts"])
    median = float(np.median(b_src))
    one_sided = np.maximum(0.0, b - median)
    p95 = lambda x: float(np.percentile(x, 95)) or 1.0
    out = {
        "current": np.asarray(res["artifact_scores"], dtype=float),
        "one-sided": one_sided,
        "residual": r,
        "combined": r / p95(r_src) + one_sided / p95(np.maximum(0.0, b_src - median)),
    }
    assert len(out["current"]) == len(cands)
    return out


def within_target(cands, scores):
    g = collections.defaultdict(lambda: ([], []))
    for c, s in zip(cands, scores):
        g[c["tid"]][1 - c["y"]].append(-s)  # lower anomaly = more typical
    wins = n = 0
    for acc, rej in g.values():
        for a in acc:
            for b in rej:
                wins += (a > b) + 0.5 * (a == b); n += 1
    return wins / max(n, 1), n


def both_labels(rows):
    labels = collections.defaultdict(set)
    for r in rows:
        labels[r["tid"]].add(r["y"])
    return [r for r in rows if labels[r["tid"]] == {0, 1}]


rng = random.Random(11)
sw, pl = load("swords_subs.jsonl"), load("prolex_subs.jsonl")
test = collections.defaultdict(lambda: ([], []))
for r in sw:
    if r["split"] == "test":
        test[r["tid"]][1 - r["y"]].append(r)
sw_cands = []
for tid, (a, j) in sorted(test.items()):
    if a and j:
        rng.shuffle(a); rng.shuffle(j)
        sw_cands += a[:4] + j[:4]
tasks = {
    "swords": ([r for r in sw if r["split"] == "dev" and r["y"] == 1], sw_cands),
    "prolex": (rng.sample([r for r in sw if r["y"] == 1], 3000), both_labels(pl)),
    "usage": (random.Random(19).sample(load("prolex_proposals_casual-v2.jsonl"), 5000), both_labels(pl)),
}
for task, (source, cands) in tasks.items():
    for variant in variants:
        h = heads(source, cands, variant)
        line = ", ".join(f"{k} {within_target(cands, v)[0]:.3f}" for k, v in h.items())
        print(f"{task} / {variant} ({within_target(cands, h['current'])[1]} pairs): {line}")
