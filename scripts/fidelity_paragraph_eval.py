"""FidelityAI on paragraph-sized rocks, with diagnostics. Local and free.

Good documents: 3 stable sentences from one held-out (test-split) article.
Editable documents: 3 sentences from one held-out article that editors later
changed. Sources (one document each): certified paragraphs; stable sentences
in threes from training articles. Reports, per source: candidates scored,
distinct scores, candidate frames whose values all fell outside the fitted
dictionary (empty after mapping), and AUC (good scored more typical).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_paragraph_eval.py <core data dir> <engine dir> [n per class = 500]
"""
import collections, random, re, sys, types

data, engine = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score, score_multi_with_rows_via_js  # noqa: E402

rng = random.Random(9)


def threes(rows):
    """Documents of 3 sentences from the same article."""
    by = collections.defaultdict(list)
    for page, sent in rows:
        by[page].append(sent)
    return [" ".join(v[i:i + 3]) for v in by.values() for i in range(0, len(v) - 2, 3)]


tsv = lambda path: [l.rstrip("\n").split("\t") for l in open(path)]
good = threes([(r[0], r[3]) for r in tsv(f"{data}/simplewiki-history/stable_test.tsv")])
editable = threes([(r[0], r[1]) for r in tsv(f"{data}/simplewiki-history/test.tsv")])
pos, neg = rng.sample(good, min(n, len(good))), rng.sample(editable, min(n, len(editable)))
cands, labels = pos + neg, [1] * len(pos) + [0] * len(neg)
words = lambda xs: sum(len(x.split()) for x in xs) / len(xs)
print(f"candidates: {len(pos)} good ({words(pos):.0f} words), {len(neg)} editable ({words(neg):.0f} words)")

sentence_count = lambda p: len(re.findall(r"[.!?](?:\s|$)", p))
certified = [r[2] for r in tsv(f"{data}/simplewiki-quality/paragraphs.tsv") if sentence_count(r[2]) >= 3]
stable_src = threes([(r[0], r[3]) for r in tsv(f"{data}/simplewiki-history/stable_train.tsv")])


def auc(scores, ys):
    p = [s for s, y in zip(scores, ys) if y]
    q = [s for s, y in zip(scores, ys) if not y]
    return sum((a > b) + 0.5 * (a == b) for a in p for b in q) / max(len(p) * len(q), 1)


for name, source in [("certified paragraphs", certified[:5000]), ("stable sentences in threes", rng.sample(stable_src, min(5000, len(stable_src))))]:
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=source, label="good"),
                             artifact_dataset=types.SimpleNamespace(episodes=cands, label="candidates"), max_qubits=20, seed=42)
    mapped = res.get("mapped_artifacts") or []
    prov = res.get("artifact_provenance") or []
    empty = sum(1 for m in mapped if not m)
    per = collections.defaultdict(list)
    for sc, pv in zip(res["artifact_scores"], prov):
        if sc is not None and isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int):
            per[pv["sourceIndex"]].append(sc)
    covered = [i for i in range(len(cands)) if per.get(i)]
    typ = [-sum(per[i]) / len(per[i]) for i in covered]
    sel = res["selected"]
    print(f"{name} ({len(source)} docs): k={sel.get('k')}, v={sel.get('v')}, qubits={sel.get('n_qubits')}; candidate frames {len(mapped)}, "
          f"empty after mapping {empty}; candidates scored {len(covered)}/{len(cands)}, distinct scores {len(set(round(x, 9) for x in typ))}; "
          f"AUC {auc(typ, [labels[i] for i in covered]):.3f}")
    # Document level (intra-, inter-tuple and congruency structure across a
    # document's frames): one score per candidate episode.
    for key in ("hier_artifact_scores", "hier_direct_artifact_scores"):
        h = res.get(key) or []
        groups = res.get("mapped_artifact_episode_groups") or []
        print(f"  {key}: {len(h)} scores for {len(cands)} candidates ({len(groups)} episode groups), {sum(1 for x in h if x is None)} empty")
        # Episodes that produced frames, in order (provenance sourceIndex).
        with_frames = sorted({pv["sourceIndex"] for pv in prov if isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int)})
        if len(with_frames) == len(h):
            ok = [(c, x) for c, x in zip(with_frames, h) if x is not None]
            vals = [-x for _, x in ok]
            print(f"    distinct {len(set(round(v, 9) for v in vals))}, AUC {auc(vals, [labels[c] for c, _ in ok]):.3f} on {len(ok)} "
                  f"(good mean {sum(-x for c, x in ok if labels[c]) / max(1, sum(labels[c] for c, _ in ok)):.4f}, "
                  f"editable mean {sum(-x for c, x in ok if not labels[c]) / max(1, sum(1 - labels[c] for c, _ in ok)):.4f})")
    print("  hier_error:", str(res.get("hier_error"))[:300])
