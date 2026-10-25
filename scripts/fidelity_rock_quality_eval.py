"""Rock quality at rock size: does a FidelityAI rulebook of 5,000 certified
passages tell certified writing from ordinary? Real data only (Simple English
Wikipedia Good / Very good articles against uncertified ones), local and free.

Passages are rock-sized: whole consecutive sentences of 100-150 characters,
3-4 lines of a rock on a phone (Title 3, 20 pt, ~36 characters per line on a
393-pt screen). Rulebook: 5,000 passages from ~70% of the certified articles.
Candidates: passages from the other certified articles against passages from
uncertified articles, the same number of each, same length band. AUC: how often
a certified passage scores more typical than an ordinary one (0.5 chance).
Writes the candidates for the judge (text_judge_eval: label, text). Baseline,
one-class like FidelityAI: mean word log-probability under the rulebook's
word counts.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_rock_quality_eval.py <simplewiki-quality dir> <engine dir> <out tsv> [rulebook = 5000]
"""
import collections, math, random, re, sys, time, types, zlib

qdir, engine, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3]
n_rulebook = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
LO, HI = 100, 150
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

split = lambda p: [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
held = lambda title: zlib.crc32(title.encode()) % 10 < 3


def passages(paragraph):
    """Non-overlapping runs of whole sentences, 100-150 characters each."""
    out, run = [], []
    for s in split(paragraph):
        run.append(s)
        text = " ".join(run)
        if len(text) > HI:
            run = [s] if len(s) <= HI else []
            text = " ".join(run)
        if LO <= len(text) <= HI:
            out.append(text); run = []
    return out


certified = [l.rstrip("\n").split("\t") for l in open(f"{qdir}/paragraphs.tsv")]
ordinary = [l.rstrip("\n").split("\t") for l in open(f"{qdir}/ordinary.tsv")]
rng = random.Random(5)
train = [x for t, _, p in certified if not held(t) for x in passages(p)]
good = [x for t, _, p in certified if held(t) for x in passages(p)]
plain = list(dict.fromkeys(x for _, p in ordinary for x in passages(p)))
rng.shuffle(train); rng.shuffle(good); rng.shuffle(plain)
rulebook = train[:n_rulebook]
n = min(len(good), len(plain), 600)
pos, neg = good[:n], plain[:n]
cands, labels = pos + neg, [1] * n + [0] * n
mean_len = lambda xs: sum(map(len, xs)) / len(xs)
print(f"rulebook: {len(rulebook)} certified passages (of {len(train)}); candidates: {n} certified held-out "
      f"({mean_len(pos):.0f} characters), {n} ordinary ({mean_len(neg):.0f} characters)")
with open(out_tsv, "w") as f:
    for y, p in zip(labels, cands):
        f.write(f"{y}\t{p}\n")


def auc(scores):
    p = [s for s, y in zip(scores, labels) if y is not None and y and s is not None]
    q = [s for s, y in zip(scores, labels) if y is not None and not y and s is not None]
    return sum((a > b) + 0.5 * (a == b) for a in p for b in q) / max(len(p) * len(q), 1)


words = lambda s: re.findall(r"[a-z']+", s.lower())
counts = collections.Counter(w for s in rulebook for w in words(s))
total, vocab = sum(counts.values()), len(counts) + 1
lp = [sum(math.log((counts[w] + 1) / (total + vocab)) for w in words(s)) / max(len(words(s)), 1) for s in cands]
print(f"rulebook's word frequencies: AUC {auc(lp):.3f}")

t = time.time()
res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=rulebook, label="certified"),
                         artifact_dataset=types.SimpleNamespace(episodes=cands, label="candidates"), max_qubits=20, seed=42)
sel, g = res["selected"], res["pipe"].grammar
print(f"FidelityAI: path {res['effective_path']}, extractor {getattr(g, 'frame_extractor', None)}, "
      f"encoding {(g.encoding or {}).get('type', 'exact index')}, k={sel.get('k')}, v={sel.get('v')}, qubits={sel.get('n_qubits')}, {time.time() - t:.0f} s")
prov = res.get("artifact_provenance") or []
per = collections.defaultdict(list)
for s, p in zip(res["artifact_scores"], prov):
    if s is not None and isinstance(p, dict) and isinstance(p.get("sourceIndex"), int):
        per[p["sourceIndex"]].append(-s)
frame = [sum(per[i]) / len(per[i]) if per.get(i) else None for i in range(len(cands))]
print(f"  frame level: {sum(x is not None for x in frame)}/{len(cands)} scored, AUC {auc(frame):.3f}")
with_frames = sorted({p["sourceIndex"] for p in prov if isinstance(p, dict) and isinstance(p.get("sourceIndex"), int)})
for key in ("hier_artifact_scores", "hier_direct_artifact_scores"):
    h = res.get(key) or []
    if len(h) == len(with_frames):
        doc = [None] * len(cands)
        for c, x in zip(with_frames, h):
            doc[c] = None if x is None else -x
        print(f"  document level ({key}): AUC {auc(doc):.3f}")
