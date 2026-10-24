"""Does FidelityAI recognize editor-certified writing? Local and free.

Source: paragraphs of Simple English Wikipedia's Good / Very good articles
(about 70% of the articles). Candidates: paragraphs of the other certified
articles (never trained on) against paragraphs of random uncertified
articles, 3+ sentences each, length-matched. AUC = how often a certified
paragraph scores more typical than an ordinary one (0.5 chance).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_certified_eval.py <simplewiki-quality dir> <engine dir> <out tsv for the judge>
"""
import collections, random, re, sys, time, types, zlib

qdir, engine, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rng = random.Random(5)
sentences = lambda p: len(re.findall(r"[.!?](?:\s|$)", p))
words = lambda p: len(p.split())
certified = [l.rstrip("\n").split("\t") for l in open(f"{qdir}/paragraphs.tsv")]  # title, category, paragraph
ordinary = [l.rstrip("\n").split("\t") for l in open(f"{qdir}/ordinary.tsv")]      # title, paragraph
held = lambda title: zlib.crc32(title.encode()) % 10 < 3  # ~30% of certified articles held out

source = [p for t, _, p in certified if not held(t) and sentences(p) >= 2]
good = [p for t, _, p in certified if held(t) and sentences(p) >= 3]
plain = [p for _, p in ordinary if sentences(p) >= 3]

# Length-match: for each good paragraph, an ordinary one from the same length band.
band = lambda p: min(words(p) // 20, 8)
by_band = collections.defaultdict(list)
for p in plain:
    by_band[band(p)].append(p)
pairs = [(g, rng.choice(by_band[band(g)])) for g in good if by_band.get(band(g))]
pos = [g for g, _ in pairs]
neg = list(dict.fromkeys(o for _, o in pairs))  # each ordinary paragraph once
cands = pos + neg
labels = [1] * len(pos) + [0] * len(neg)
mean = lambda xs: sum(map(words, xs)) / max(len(xs), 1)
print(f"source: {len(source)} certified paragraphs; candidates: {len(pos)} certified held-out ({mean(pos):.0f} words), "
      f"{len(neg)} ordinary ({mean(neg):.0f} words)")
with open(out_tsv, "w") as f:
    for y, p in zip(labels, cands):
        f.write(f"{y}\t{p}\n")


def auc(scores, ys):
    p = [s for s, y in zip(scores, ys) if y]
    q = [s for s, y in zip(scores, ys) if not y]
    return sum((a > b) + 0.5 * (a == b) for a in p for b in q) / max(len(p) * len(q), 1)


t = time.time()
res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=source, label="certified"),
                         artifact_dataset=types.SimpleNamespace(episodes=cands, label="candidates"), max_qubits=20, seed=42)
sel = res["selected"]
grammar = res["pipe"].grammar
print(f"FidelityAI: path {res['effective_path']}, extractor {getattr(grammar, 'frame_extractor', None)}, "
      f"encoding {(getattr(grammar, 'encoding', None) or {}).get('type', 'exact index')}, k={sel.get('k')}, v={sel.get('v')}, qubits={sel.get('n_qubits')}, {time.time() - t:.0f} s")
prov = res.get("artifact_provenance") or []
per = collections.defaultdict(list)
for sc, pv in zip(res["artifact_scores"], prov):
    if sc is not None and isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int):
        per[pv["sourceIndex"]].append(sc)
covered = [i for i in range(len(cands)) if per.get(i)]
print(f"  frame level: {len(covered)}/{len(cands)} scored, AUC {auc([-sum(per[i]) / len(per[i]) for i in covered], [labels[i] for i in covered]):.3f}")
with_frames = sorted({pv["sourceIndex"] for pv in prov if isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int)})
for key in ("hier_artifact_scores", "hier_direct_artifact_scores"):
    h = res.get(key) or []
    if len(h) == len(with_frames):
        ok = [(c, x) for c, x in zip(with_frames, h) if x is not None]
        print(f"  document level ({key}): AUC {auc([-x for _, x in ok], [labels[c] for c, _ in ok]):.3f} on {len(ok)}, "
              f"{len(set(round(x, 9) for _, x in ok))} distinct")
