"""Does FidelityAI recognize a good rock? Local and free.

Candidates (held-out test-split articles): stable sentences (unchanged for 20+
revisions and a year+: editors left them alone) against sentences editors
later changed (the "before" side of their edits). AUC = how often a stable
sentence scores more typical than a changed one (0.5 chance). Sources (one
document each, path B): editor-certified paragraphs (Good / Very good
articles), and stable sentences from training articles.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_quality_eval.py <core data dir> <engine dir> [n per class = 1000]
Writes <data>/fidelity-edits/quality_candidates.tsv (label, sentence) for the judge baseline.
"""
import collections, random, sys, time, types

data, engine = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rng = random.Random(7)
stable_test = [l.rstrip("\n").split("\t")[3] for l in open(f"{data}/simplewiki-history/stable_test.tsv")]
changed_test = [l.rstrip("\n").split("\t")[1] for l in open(f"{data}/simplewiki-history/test.tsv")]
# Sources need several frames per document (a document shorter than k rules
# that k out for the whole fit): certified paragraphs of 3+ sentences, and
# stable sentences in threes from the same article.
import re  # noqa: E402
sentence_count = lambda p: len(re.findall(r"[.!?](?:\s|$)", p))
certified = [l.rstrip("\n").split("\t")[2] for l in open(f"{data}/simplewiki-quality/paragraphs.tsv")]
certified = [p for p in certified if sentence_count(p) >= 3]
by_article = collections.defaultdict(list)
for l in open(f"{data}/simplewiki-history/stable_train.tsv"):
    r = l.rstrip("\n").split("\t")
    by_article[r[0]].append(r[3])
stable_train = [" ".join(v[i:i + 3]) for v in by_article.values() for i in range(0, len(v) - 2, 3)]

# Same number per class; changed sentences drawn to match the stable ones' lengths.
pos = rng.sample(stable_test, min(n, len(stable_test)))
by_len = collections.defaultdict(list)
for s in changed_test:
    by_len[min(len(s.split()) // 5, 12)].append(s)
neg = []
for s in pos:
    bucket = by_len.get(min(len(s.split()) // 5, 12)) or changed_test
    neg.append(rng.choice(bucket))
cands = pos + neg
labels = [1] * len(pos) + [0] * len(neg)
mean_len = lambda xs: sum(len(x.split()) for x in xs) / len(xs)
print(f"candidates: {len(pos)} stable (mean {mean_len(pos):.1f} words), {len(neg)} later changed (mean {mean_len(neg):.1f} words)")
with open(f"{data}/fidelity-edits/quality_candidates.tsv", "w") as f:
    for y, s in zip(labels, cands):
        f.write(f"{y}\t{s}\n")


def auc(scores, ys):
    p = [s for s, y in zip(scores, ys) if y]
    q = [s for s, y in zip(scores, ys) if not y]
    return sum((a > b) + 0.5 * (a == b) for a in p for b in q) / max(len(p) * len(q), 1)


for name, source in [("certified paragraphs", certified[:5000]), ("stable sentences", rng.sample(stable_train, min(5000, len(stable_train))))]:
    t = time.time()
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=source, label="good"),
                             artifact_dataset=types.SimpleNamespace(episodes=cands, label="candidates"), max_qubits=20, seed=42)
    per = collections.defaultdict(list)
    for sc, pv in zip(res["artifact_scores"], res.get("artifact_provenance") or []):
        if sc is not None and isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int):
            per[pv["sourceIndex"]].append(sc)
    covered = [i for i in range(len(cands)) if per.get(i)]
    # Distance from the fit: larger = less typical.
    typ = [-sum(per[i]) / len(per[i]) for i in covered]
    ys = [labels[i] for i in covered]
    ties = len(typ) - len(set(typ))
    sel = res["selected"]
    print(f"FidelityAI, source = {name} ({len(source)} documents): path {res['effective_path']}, k={sel.get('k')}, qubits={sel.get('n_qubits')}, "
          f"{len(covered)}/{len(cands)} scored, {ties} tied scores, AUC {auc(typ, ys):.3f}, {time.time() - t:.0f} s")
