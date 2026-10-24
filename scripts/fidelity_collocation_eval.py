"""Usage text as the rulebook, ProLex as the candidates: do attested word
pairings tell an acceptable substitute from an unacceptable one? Real text
and real human judgments only; ProLex is scored, never in a rulebook.

Input: examples/collocations.rs output (<prefix>.candidates.jsonl,
<prefix>.rulebook.tsv). Reported within each target (how often an acceptable
substitute beats an unacceptable one; 0.5 chance), for corpus counts (the
free ceiling: raw count and PMI, pointwise mutual information, of the
substitute's pairings) and for FidelityAI, whose rulebook is corpus triples
(governor, relation, dependent) sampled by frequency, sharing a candidate's
governor and relation or relation and dependent.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_collocation_eval.py <prefix> <engine dir> [rulebook records = 5000]
"""
import collections, json, math, random, sys, time, types

prefix, engine = sys.argv[1], sys.argv[2]
n_rulebook = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
cands = [json.loads(l) for l in open(f"{prefix}.candidates.jsonl")]
groups = collections.defaultdict(set)
for c in cands:
    groups[c["tid"]].add(c["y"])
cands = [c for c in cands if groups[c["tid"]] == {0, 1}]


def within_target(scores):
    g = collections.defaultdict(lambda: ([], []))
    for c, s in zip(cands, scores):
        if s is not None:
            g[c["tid"]][1 - c["y"]].append(s)
    wins = n = 0
    for acc, rej in g.values():
        for a in acc:
            for b in rej:
                wins += (a > b) + 0.5 * (a == b); n += 1
    return wins / max(n, 1), n


def triples(c, side):
    return ([c["up"]] if c["up"] and side in ("up", "both") else []) + (c["down"] if side in ("down", "both") else [])


def pmi(t, tokens):
    # log P(gov, rel, dep) / (P(gov, rel) P(dep)), smoothed.
    return math.log((t["count"] + 0.5) * tokens / ((t["gov_rel"] + 1) * (t["dep_n"] + 1)))


print(f"candidates: {len(cands)} ProLex substitutes ({sum(c['y'] for c in cands)} acceptable) for {len({c['tid'] for c in cands})} targets; "
      f"with a governor pairing: {sum(1 for c in cands if c['up'])}; attested there: {sum(1 for c in cands if c['up'] and c['up']['count'])}")
for side in ("up", "down", "both"):
    count = [sum(math.log1p(t["count"]) for t in triples(c, side)) if triples(c, side) else None for c in cands]
    p = [sum(pmi(t, c["tokens"]) for t in triples(c, side)) / len(triples(c, side)) if triples(c, side) else None for c in cands]
    print(f"  counts ({side}): log count {within_target(count)[0]:.3f}, PMI {within_target(p)[0]:.3f} of {within_target(p)[1]} pairs")
# How common the substitute itself is (no pairing): the baseline the counts must beat.
freq = [math.log1p(c["up"]["dep_n"]) if c["up"] else (math.log1p(c["down"][0]["gov_n"]) if c["down"] else None) for c in cands]
print(f"  substitute's own frequency: {within_target(freq)[0]:.3f}")
# Pairing count given the substitute's frequency: log count minus log frequency (P(pairing | substitute)).
cond = [math.log1p(c["up"]["count"]) - math.log1p(c["up"]["dep_n"]) if c["up"] else None for c in cands]
print(f"  pairing given the substitute (up): {within_target(cond)[0]:.3f} of {within_target(cond)[1]} pairs")

sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rng = random.Random(13)
rows = [l.rstrip("\n").split("\t") for l in open(f"{prefix}.rulebook.tsv")]
# Candidate keys only: a candidate's governor and relation (up), or relation
# and dependent (down); triples sampled within each key by corpus frequency.
up_keys = {(c["up"]["gov"], c["up"]["rel"]) for c in cands if c["up"]}
down_keys = {(t["rel"], t["dep"]) for c in cands for t in c["down"]}
by_key = collections.defaultdict(list)
for g, r, d, n in rows:
    if (g, r) in up_keys:
        by_key[("up", g, r)].append(((g, r, d), int(n)))
    if (r, d) in down_keys:
        by_key[("down", r, d)].append(((g, r, d), int(n)))
keys = sorted(by_key)
per_key = max(1, n_rulebook // max(len(keys), 1))
source = []
for k in keys:
    ts, weights = zip(*by_key[k])
    source += [{"gov": g, "rel": r, "dep": d} for g, r, d in rng.choices(ts, weights=weights, k=per_key)]
rng.shuffle(source)
source = source[:n_rulebook]
records, owner = [], []
for i, c in enumerate(cands):
    for t in triples(c, "both"):
        records.append({"gov": t["gov"], "rel": t["rel"], "dep": t["dep"]}); owner.append(i)
print(f"FidelityAI rulebook: {len(source)} corpus triples ({per_key} per candidate key of {len(keys)}, by frequency); {len(records)} candidate triples")
t0 = time.time()
res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=source, label="usage"),
                         artifact_dataset=types.SimpleNamespace(episodes=records, label="prolex"), max_qubits=20, seed=42)
sel, gr = res["selected"], res["pipe"].grammar
scores = res["artifact_scores"]
per = collections.defaultdict(list)
for i, s in zip(owner, scores):
    if s is not None:
        per[i].append(-s)
fid = [sum(per[i]) / len(per[i]) if per.get(i) else None for i in range(len(cands))]
print(f"  FidelityAI: path {res['effective_path']}, encoding {(getattr(gr, 'encoding', None) or {}).get('type', 'exact index')}, "
      f"k={sel.get('k')}, qubits={sel.get('n_qubits')}, {time.time() - t0:.0f} s; accepted beats rejected {within_target(fid)[0]:.3f} of {within_target(fid)[1]} pairs")
