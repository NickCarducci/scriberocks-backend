"""Style conformance: does FidelityAI recognise a writer's own prose? Real
data only (Tatoeba sentences with their contributors, CC-BY 2.0 FR), local
and free. For each of several prolific contributors: rulebook = 2,000 of
their sentences (prose, path B); candidates = 300 held-out sentences by them
and 300 by other contributors. AUC: how often the author's own sentence
scores more typical (0.5 chance). Baseline, also one-class: mean word
log-probability under the author's own word counts (add-one smoothing).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_author_eval.py <eng_sentences_detailed.tsv.bz2> <engine dir> [authors]
"""
import bz2, collections, math, random, re, sys, time, types

path, engine = sys.argv[1], sys.argv[2]
authors = sys.argv[3].split(",") if len(sys.argv) > 3 else ["CK", "Amastan", "OsoHombre", "CH", "Hybrid", "CM"]
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

by_author = collections.defaultdict(list)
for line in bz2.open(path, "rt"):
    c = line.rstrip("\n").split("\t")
    if len(c) >= 4 and 5 <= len(c[2].split()) <= 25:
        by_author[c[3]].append(c[2])
words = lambda s: re.findall(r"[a-z']+", s.lower())


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


for author in authors:
    rng = random.Random(7)
    own = list(dict.fromkeys(by_author[author])); rng.shuffle(own)
    others = [s for a, ss in by_author.items() if a != author for s in ss]
    rulebook, held, foreign = own[:2000], own[2000:2300], rng.sample(others, 300)
    counts = collections.Counter(w for s in rulebook for w in words(s))
    total, vocab = sum(counts.values()), len(counts) + 1
    lp = lambda s: sum(math.log((counts[w] + 1) / (total + vocab)) for w in words(s)) / max(len(words(s)), 1)
    t = time.time()
    cands = held + foreign
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=rulebook, label=author),
                             artifact_dataset=types.SimpleNamespace(episodes=cands, label="candidates"), max_qubits=20, seed=42)
    per = collections.defaultdict(list)
    for s, p in zip(res["artifact_scores"], res.get("artifact_provenance") or []):
        if s is not None and isinstance(p, dict) and "sourceIndex" in p:
            per[p["sourceIndex"]].append(-s)
    fid = {i: sum(v) / len(v) for i, v in per.items()}
    pos = [fid[i] for i in range(300) if i in fid]
    neg = [fid[i] for i in range(300, 600) if i in fid]
    print(f"{author}: FidelityAI AUC {auc(pos, neg):.3f} ({len(pos)}+{len(neg)} scored, k={res['selected'].get('k')}, {time.time() - t:.0f} s); "
          f"author's word frequencies AUC {auc([lp(s) for s in held], [lp(s) for s in foreign]):.3f}", flush=True)
