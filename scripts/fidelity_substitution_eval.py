"""Word fit learned from human judgments: a FidelityAI rulebook of accepted
substitutions (SWORDS dev), scoring SWORDS test substitutes. Real data only,
local and free.

Records (examples/substitution_records.rs): the original word's meaning in
its sentence, the substitute's meaning in place, part of speech, relation,
and the governor. Variants:
  words     target, substitute and governor lemmas (no meanings)
  category  meanings as lexicographer categories (noun.artifact, ...)
  synset    meanings as synsets
  relation  how the substitute's meaning relates to the original's (same
            synset, same category, other, unknown), with the grammar and the
            original meaning's category
Also scored, for comparison: a plain rule (the substitute's meaning is the
original's: same synset, else same category) and the casual judge (run
separately on the candidates file this writes).
Reported: within each target, how often an accepted substitute scores more
typical than a rejected one (0.5 chance).

usage: [CANDIDATES=<records jsonl>] ~/.venvs/fidelityai/bin/python -I fidelity_substitution_eval.py <swords_subs.jsonl> <engine dir> [variants] [per target = 4]
"""
import collections, json, random, sys, time, types

path, engine = sys.argv[1], sys.argv[2]
variants = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else ["words", "category", "synset"]
per_target = int(sys.argv[4]) if len(sys.argv) > 4 else 4
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rows = [json.loads(l) for l in open(path)]
source_rows = [r for r in rows if r["split"] == "dev" and r["y"] == 1]
rng = random.Random(11)
# CANDIDATES=<records jsonl>: score another set (e.g. ProLex, scored only,
# never in a rulebook) against the SWORDS dev rulebooks.
import os  # noqa: E402
other = os.environ.get("CANDIDATES")
by_target = collections.defaultdict(lambda: ([], []))
for r in ([json.loads(l) for l in open(other)] if other else rows):
    if other or r["split"] == "test":
        by_target[r["tid"]][1 - r["y"]].append(r)
cands = []
for tid, (acc, rej) in sorted(by_target.items()):
    if acc and rej:
        rng.shuffle(acc); rng.shuffle(rej)
        cands += acc[:per_target] + rej[:per_target]
print(f"rulebook: {len(source_rows)} accepted dev substitutions; candidates: {len(cands)} test substitutions "
      f"({sum(r['y'] for r in cands)} accepted) for {len({r['tid'] for r in cands})} targets")
with open((other or path).replace(".jsonl", ".candidates.tsv"), "w") as f:
    for r in cands:
        f.write(f"{r['tid']}\t{r['y']}\t{r['text']}\n")


def within_target(scores):
    groups = collections.defaultdict(lambda: ([], []))
    for r, s in zip(cands, scores):
        if s is not None:
            groups[r["tid"]][1 - r["y"]].append(s)
    wins = n = 0
    for acc, rej in groups.values():
        for a in acc:
            for b in rej:
                wins += (a > b) + 0.5 * (a == b); n += 1
    return wins / max(n, 1), n


def m(r, key, level):
    x = r.get(key)
    return x[level] if x else None


rule = [2 if m(r, "meaning", "s") and m(r, "meaning", "s") == m(r, "sub_meaning", "s")
        else 1 if m(r, "meaning", "c") and m(r, "meaning", "c") == m(r, "sub_meaning", "c") else 0 for r in cands]
print("rule (substitute's meaning is the original's): accepted beats rejected %.3f of %d pairs" % within_target(rule))


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


# A second rulebook of rejected dev substitutions (raters' rejections), as
# many as the accepted: a candidate's fit is how much more typical it is of
# the accepted rulebook than of the rejected one.
rejected_rows = rng.sample([r for r in rows if r["split"] == "dev" and r["y"] == 0], len(source_rows))


def fit(source, variant, label):
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[record(r, variant) for r in source], label=label),
                             artifact_dataset=types.SimpleNamespace(episodes=[record(r, variant) for r in cands], label="swords-test"),
                             max_qubits=20, seed=42)
    sel, g = res["selected"], res["pipe"].grammar
    desc = f"path {res['effective_path']}, encoding {(getattr(g, 'encoding', None) or {}).get('type', 'exact index')}, k={sel.get('k')}, qubits={sel.get('n_qubits')}"
    scores = res["artifact_scores"]
    return (scores if len(scores) == len(cands) else None), desc


for variant in variants:
    t = time.time()
    acc, desc = fit(source_rows, variant, "accepted")
    rej, _ = fit(rejected_rows, variant, "rejected")
    if acc is None or rej is None:
        print(f"{variant}: scores don't match the candidates; skipped")
        continue
    one = within_target([None if s is None else -s for s in acc])
    two = within_target([None if a is None or b is None else b - a for a, b in zip(acc, rej)])
    print(f"{variant}: {desc}, {time.time() - t:.0f} s; accepted beats rejected: accepted rulebook {one[0]:.3f}, "
          f"accepted against rejected rulebooks {two[0]:.3f} (of {two[1]} pairs)")
