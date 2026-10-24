"""Slot records: the pairing a substitute makes, nothing else. Each record is
(governor, relation, substitute), as words or as WordNet meanings (synsets;
a word without one keeps its lemma), so two substitutes for the same slot
differ in one value of three. Real data and real human judgments only;
ProLex is scored, never in a rulebook from ProLex raters' labels.

Tasks (within target; accepted / acceptable beats rejected / unacceptable):
  swords   rulebook SWORDS dev accepted          → SWORDS test
  prolex   rulebook SWORDS accepted (all)        → ProLex
  usage    rulebook casual v2's proposals for ProLex slots → ProLex
           (local only: ProLex sentences are ETS-licensed)
Reported: FidelityAI's primary score, and a plain count of the record in the
rulebook (the free reference).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_slot_eval.py <core data dir> <engine dir>
"""
import collections, json, random, sys, types

data, engine = sys.argv[1], sys.argv[2]
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

load = lambda p: [json.loads(l) for l in open(f"{data}/fidelity-edits/{p}")]


def slot(r, level):
    if level == "words":
        out = {"governor": r["governor"], "rel": r["rel"], "substitute": r["substitute"]}
    else:
        m = lambda key, lemma: (r.get(key) or {}).get("s") or lemma
        out = {"governor": m("gov_meaning", r["governor"]), "rel": r["rel"], "substitute": m("sub_meaning", r["substitute"])}
    return {k: v for k, v in out.items() if v is not None}


def within_target(cands, typicality):
    g = collections.defaultdict(lambda: ([], []))
    for c, s in zip(cands, typicality):
        if s is not None:
            g[c["tid"]][1 - c["y"]].append(s)
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
    "prolex": ([r for r in sw if r["y"] == 1], both_labels(pl)),
    "usage": (random.Random(19).sample(load("prolex_proposals_casual-v2.jsonl"), 5000), both_labels(pl)),
}
for task, (source, cands) in tasks.items():
    for level in ("words", "meanings"):
        src = [slot(r, level) for r in source]
        cnd = [slot(r, level) for r in cands]
        key = lambda d: tuple(sorted(d.items()))
        counts = collections.Counter(key(d) for d in src)
        count_rate = within_target(cands, [counts[key(d)] for d in cnd])[0]
        res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=src, label="rulebook"),
                                 artifact_dataset=types.SimpleNamespace(episodes=cnd, label="candidates"), max_qubits=20, seed=42)
        g = res["pipe"].grammar
        rate, n = within_target(cands, [None if s is None else -s for s in res["artifact_scores"]])
        seen = sum(1 for d in cnd if counts[key(d)])
        print(f"{task} / {level}: {len(src)} rulebook records, {seen}/{len(cnd)} candidates seen in it; "
              f"encoding {(g.encoding or {}).get('type', 'exact index')}, k={res['selected'].get('k')}; "
              f"FidelityAI {rate:.3f}, count {count_rate:.3f} of {n} pairs")
