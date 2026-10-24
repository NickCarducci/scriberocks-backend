"""The usage model builds the source; ProLex tests it. FidelityAI inputs:
source = the usage model's own proposals for each ProLex slot
(examples/usage_proposals.rs → substitution_records.rs); candidates = the
human-judged ProLex substitutes (acceptable / unacceptable) in the same slots.
Local evaluation only: ProLex sentences are ETS-licensed, so the fitted
models are discarded and never shipped.

Reported within each ProLex target (how often an acceptable substitute beats
an unacceptable one; 0.5 chance), for FidelityAI's record variants and for the
usage model's own ranking (a candidate it proposed, by rank; unproposed last).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_usage_source_eval.py <proposals.jsonl> <prolex_subs.jsonl> <engine dir> [variants] [source records = 5000]
"""
import collections, json, random, sys, time, types

prop_path, pl_path, engine = sys.argv[1], sys.argv[2], sys.argv[3]
variants = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else ["words", "category", "synset", "relation"]
n_source = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

proposals = [json.loads(l) for l in open(prop_path)]
labels = collections.defaultdict(set)
pl = [json.loads(l) for l in open(pl_path)]
for r in pl:
    labels[r["tid"]].add(r["y"])
cands = [r for r in pl if labels[r["tid"]] == {0, 1}]


def within_target(scores):
    g = collections.defaultdict(lambda: ([], []))
    for r, s in zip(cands, scores):
        if s is not None:
            g[r["tid"]][1 - r["y"]].append(s)
    wins = n = 0
    for acc, rej in g.values():
        for a in acc:
            for b in rej:
                wins += (a > b) + 0.5 * (a == b); n += 1
    return wins / max(n, 1), n


rank = collections.defaultdict(dict)
for r in proposals:
    rank[r["tid"]].setdefault(r["substitute"], len(rank[r["tid"]]))
proposed = [-rank[r["tid"]][r["substitute"]] if r["substitute"] in rank[r["tid"]] else -99 for r in cands]
print(f"source: {len(proposals)} usage-model proposals for {len(rank)} slots; candidates: {len(cands)} ProLex substitutes "
      f"({sum(r['y'] for r in cands)} acceptable) for {len({r['tid'] for r in cands})} targets; "
      f"{sum(1 for s in proposed if s > -99)} of them among the model's proposals")
print("usage model's own ranking: acceptable beats unacceptable %.3f of %d pairs" % within_target(proposed))


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
        level = "c" if variant == "category" else "s"
        out.update(target=r["target"], meaning=m(r, "meaning", level), sub_meaning=m(r, "sub_meaning", level), gov_meaning=m(r, "gov_meaning", level))
    return {k: v for k, v in out.items() if v is not None}


source_rows = random.Random(19).sample(proposals, min(n_source, len(proposals)))
for variant in variants:
    t = time.time()
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[record(r, variant) for r in source_rows], label="usage-model"),
                             artifact_dataset=types.SimpleNamespace(episodes=[record(r, variant) for r in cands], label="prolex"),
                             max_qubits=20, seed=42)
    g = res["pipe"].grammar
    rate, n = within_target([None if s is None else -s for s in res["artifact_scores"]])
    print(f"{variant}: columns {(g.columns or {}).get('kept')}, encoding {(g.encoding or {}).get('type', 'exact index')}, k={res['selected'].get('k')}, "
          f"{time.time() - t:.0f} s; acceptable beats unacceptable {rate:.3f} of {n} pairs")
