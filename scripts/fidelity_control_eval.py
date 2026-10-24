"""Two human-labeled corpora, different humans: SWORDS raters judged each
substitute in published prose; ProLex raters judged substitutes in learners'
essays. Candidates: ProLex substitutes that SWORDS raters accepted for the
same word in another sentence (each passes as a word pair), labeled by ProLex
raters in this sentence; the rejected ones fail only in context. Rulebooks:
every SWORDS substitution (accepted, and as many rejected), never ProLex.

Reported within each ProLex target (how often an accepted substitute beats a
rejected one; 0.5 chance), for FidelityAI's record variants, and the
context-free baseline over all ProLex judgments SWORDS also judged (the SWORDS
verdict carried over: share of SWORDS contexts that accepted the pair).
Writes the candidates for the judge (text_judge_eval).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_control_eval.py <swords_subs.jsonl> <prolex_subs.jsonl> <engine dir> [variants]
"""
import collections, json, random, sys, time, types

sw_path, pl_path, engine = sys.argv[1], sys.argv[2], sys.argv[3]
variants = sys.argv[4].split(",") if len(sys.argv) > 4 else ["words", "category", "synset", "relation"]
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

sw = [json.loads(l) for l in open(sw_path)]
pl = [json.loads(l) for l in open(pl_path)]
verdicts = collections.defaultdict(list)
for r in sw:
    verdicts[(r["target"], r["substitute"])].append(r["y"])
share = lambda r: sum(verdicts[(r["target"], r["substitute"])]) / len(verdicts[(r["target"], r["substitute"])])


def within_target(rows, scores):
    g = collections.defaultdict(lambda: ([], []))
    for r, s in zip(rows, scores):
        if s is not None:
            g[r["tid"]][1 - r["y"]].append(s)
    wins = n = 0
    for acc, rej in g.values():
        for a in acc:
            for b in rej:
                wins += (a > b) + 0.5 * (a == b); n += 1
    return wins / max(n, 1), n


def with_both(rows):
    labels = collections.defaultdict(set)
    for r in rows:
        labels[r["tid"]].add(r["y"])
    return [r for r in rows if labels[r["tid"]] == {0, 1}]


judged = with_both([r for r in pl if (r["target"], r["substitute"]) in verdicts])
print("context-free (SWORDS verdict carried over): accepted beats rejected %.3f of %d pairs" % within_target(judged, [share(r) for r in judged]))
cands = with_both([r for r in pl if (r["target"], r["substitute"]) in verdicts and share(r) > 0.5])
print(f"candidates: {len(cands)} ProLex substitutes SWORDS raters accepted elsewhere ({sum(r['y'] for r in cands)} accepted by ProLex raters here) "
      f"for {len({r['tid'] for r in cands})} targets")
with open(pl_path.replace(".jsonl", ".control.tsv"), "w") as f:
    for r in cands:
        f.write(f"{r['tid']}\t{r['y']}\t{r['text']}\n")

rng = random.Random(17)
accepted = [r for r in sw if r["y"] == 1]
rejected = rng.sample([r for r in sw if r["y"] == 0], len(accepted))


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
        out.update(meaning=m(r, "meaning", level), sub_meaning=m(r, "sub_meaning", level), gov_meaning=m(r, "gov_meaning", level))
    return {k: v for k, v in out.items() if v is not None}


def fit(source, variant):
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[record(r, variant) for r in source], label="swords"),
                             artifact_dataset=types.SimpleNamespace(episodes=[record(r, variant) for r in cands], label="prolex"),
                             max_qubits=20, seed=42)
    return res["artifact_scores"], res


print(f"rulebooks: {len(accepted)} accepted and {len(rejected)} rejected SWORDS substitutions")
for variant in variants:
    t = time.time()
    acc, res = fit(accepted, variant)
    rej, _ = fit(rejected, variant)
    one = within_target(cands, [-s for s in acc])
    two = within_target(cands, [b - a for a, b in zip(acc, rej)])
    print(f"{variant}: columns {res['pipe'].grammar.columns['kept'] if res['pipe'].grammar.columns else None}, k={res['selected'].get('k')}, "
          f"{time.time() - t:.0f} s; accepted rulebook {one[0]:.3f}, accepted against rejected {two[0]:.3f} of {two[1]} pairs")
