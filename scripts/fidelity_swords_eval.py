"""Modularly passing, compositionally failing: does FidelityAI tell a fitting
substitute from a synonym that doesn't fit this sentence? Real data only.

Rulebook (source): stable Simple English Wikipedia sentences (proper use).
Candidates: SWORDS test sentences (published prose; human-rated substitutes)
with each judged substitute in place of the target: accepted (> 50% of raters)
and rejected (no rater accepted) — the sentences raters actually judged.
Reported: within each target, how often an accepted version scores more
typical than a rejected one (0.5 chance). Writes the candidates for the judge.

usage: [VISIBLE=<targets file>] ~/.venvs/fidelityai/bin/python -I fidelity_swords_eval.py <core data dir> <engine dir> [targets = 400]
VISIBLE: count only targets whose substitutes change the scribe-v1 frames.
"""
import collections, json, random, sys, time, types

data, engine = sys.argv[1], sys.argv[2]
n_targets = int(sys.argv[3]) if len(sys.argv) > 3 else 400
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rng = random.Random(3)
d = json.load(open(f"{data}/swords/swords-v1.1_test.json"))
labels_of = {sid: [l for l in d["substitute_labels"].get(sid, []) if l != "UNSURE"] for sid in d["substitutes"]}
by_target = collections.defaultdict(lambda: ([], []))
for sid, sub in d["substitutes"].items():
    ls = labels_of[sid]
    w = sub["substitute"]
    if not ls or " " in w:
        continue
    yes = sum(l.startswith("TRUE") for l in ls) / len(ls)
    acc, rej = by_target[sub["target_id"]]
    (acc if yes > 0.5 else rej if yes == 0 else []).append(w)


def sentence_with(tid, word):
    t = d["targets"][tid]
    ctx = d["contexts"][t["context_id"]]["context"]
    off, tw = t["offset"], t["target"]
    ends = lambda i: ctx[i] == "\n" or (ctx[i] in ".!?" and (i + 1 >= len(ctx) or ctx[i + 1].isspace()))
    start = next((i + 1 for i in range(off - 1, -1, -1) if ends(i)), 0)
    end = next((i + 1 for i in range(off + len(tw), len(ctx)) if ends(i)), len(ctx))
    s = ctx[start:off] + word + ctx[off + len(tw):end]
    return " ".join(s.split())


targets = [tid for tid, (a, r) in by_target.items() if a and r]
rng.shuffle(targets)
targets = targets[:n_targets]
cands, meta = [], []  # meta: (target, accepted?)
for tid in targets:
    acc, rej = by_target[tid]
    for w in acc[:2]:
        cands.append(sentence_with(tid, w)); meta.append((tid, 1))
    for w in rej[:2]:
        cands.append(sentence_with(tid, w)); meta.append((tid, 0))
with open(f"{data}/fidelity-edits/swords_candidates.tsv", "w") as f:
    for (tid, y), s in zip(meta, cands):
        f.write(f"{tid}\t{y}\t{s}\n")

source = []
for line in open(f"{data}/simplewiki-history/stable_train.tsv"):
    source.append(line.rstrip("\n").split("\t")[3])
source = rng.sample(source, 5000)
print(f"rulebook: {len(source)} stable sentences; candidates: {len(cands)} SWORDS sentences "
      f"({sum(y for _, y in meta)} accepted, {sum(1 - y for _, y in meta)} rejected) for {len(targets)} targets")


import os  # noqa: E402
VISIBLE = set(open(os.environ["VISIBLE"]).read().split()) if os.environ.get("VISIBLE") else None


def within_target(scores):
    """How often an accepted version beats a rejected one of the same target
    (scores: typicality, higher = more typical; None skipped)."""
    groups = collections.defaultdict(lambda: ([], []))
    for (tid, y), s in zip(meta, scores):
        if s is not None and (VISIBLE is None or tid in VISIBLE):
            groups[tid][1 - y].append(s)
    wins = n = 0
    for acc, rej in groups.values():
        for a in acc:
            for r in rej:
                wins += (a > r) + 0.5 * (a == r); n += 1
    return wins / max(n, 1), n


t = time.time()
res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=source, label="proper-use"),
                         artifact_dataset=types.SimpleNamespace(episodes=cands, label="swords"), max_qubits=20, seed=42)
sel, g = res["selected"], res["pipe"].grammar
print(f"FidelityAI: path {res['effective_path']}, extractor {getattr(g, 'frame_extractor', None)}, "
      f"encoding {(getattr(g, 'encoding', None) or {}).get('type', 'exact index')}, k={sel.get('k')}, v={sel.get('v')}, qubits={sel.get('n_qubits')}, {time.time() - t:.0f} s")
prov = res.get("artifact_provenance") or []
per = collections.defaultdict(list)
for sc, pv in zip(res["artifact_scores"], prov):
    if sc is not None and isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int):
        per[pv["sourceIndex"]].append(sc)
frame = [(-sum(per[i]) / len(per[i]) if per.get(i) else None) for i in range(len(cands))]
print("  frame level: accepted beats rejected %.3f of %d pairs" % within_target(frame))
with_frames = sorted({pv["sourceIndex"] for pv in prov if isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int)})
for key in ("hier_artifact_scores", "hier_direct_artifact_scores"):
    h = res.get(key) or []
    if len(h) == len(with_frames):
        doc = [None] * len(cands)
        for c, x in zip(with_frames, h):
            doc[c] = None if x is None else -x
        print(f"  document level ({key}): accepted beats rejected %.3f of %d pairs" % within_target(doc))
