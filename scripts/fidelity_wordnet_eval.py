"""Generalising human-judged substitutes with WordNet (Open English WordNet:
lexicographers' work, a separate human-built corpus). Slot records
(governor, relation, substitute) as in fidelity_slot_eval.py, plus:

  words      the words (baseline)
  meanings   synsets from the sense head (baseline)
  parent     the substitute's synset replaced by its parent: first hypernym
             (nouns, verbs), or the head of its similar-adjective cluster;
             a synset with neither keeps itself
  meanings+parent  both, four columns
  synonyms   words, with the rulebook expanded by WordNet: each record also
             appears with every other one-word member of its substitute's
             synset (as the sense head read it)

Tasks (within target; accepted / acceptable beats rejected / unacceptable):
  swords   rulebook SWORDS dev accepted → SWORDS test
  prolex   rulebook SWORDS accepted (all) → ProLex
  usage    rulebook casual v2's proposals for ProLex slots → ProLex
           (local only: ProLex sentences are ETS-licensed)
Reported per variant: rulebook size, candidates whose record the rulebook
contains, FidelityAI's primary score and a plain count of the record.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_wordnet_eval.py <core data dir> <engine dir> [variants]
"""
import collections, json, random, sys, types

data, engine = sys.argv[1], sys.argv[2]
variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["words", "meanings", "parent", "meanings+parent", "synonyms"]
CAP = {"meanings+parent": 4000}  # four columns: k = 4, 65,536 features; others k = 3
DEFAULT_CAP = 20000
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

load = lambda p: [json.loads(l) for l in open(f"{data}/fidelity-edits/{p}")]
wordnet = {s["id"]: s for s in load("wordnet.jsonl")}


def parent(sid):
    s = wordnet.get(sid)
    if not s:
        return sid
    return (s["hypernyms"] or s["similar"] or [sid])[0]


def syn(r, key):
    return (r.get(key) or {}).get("s")


def records(r, variant):
    gov_s, sub_s = syn(r, "gov_meaning"), syn(r, "sub_meaning")
    base = {"rel": r["rel"]}
    if variant in ("words", "synonyms"):
        out = [{**base, "governor": r["governor"], "substitute": r["substitute"]}]
    elif variant == "meanings":
        out = [{**base, "governor": gov_s or r["governor"], "substitute": sub_s or r["substitute"]}]
    elif variant == "parent":
        out = [{**base, "governor": gov_s or r["governor"], "substitute": parent(sub_s) if sub_s else r["substitute"]}]
    else:  # meanings+parent
        out = [{**base, "governor": gov_s or r["governor"], "substitute": sub_s or r["substitute"],
                "parent": parent(sub_s) if sub_s else r["substitute"]}]
    return [{k: v for k, v in o.items() if v is not None} for o in out]


def rulebook(rows, variant, rng):
    out = [rec for r in rows for rec in records(r, variant)]
    if variant == "synonyms":
        for r in rows:
            s = wordnet.get(syn(r, "sub_meaning") or "")
            for m in (s or {}).get("members", []):
                if " " not in m and m != r["substitute"]:
                    out.append({"rel": r["rel"], "governor": r["governor"], "substitute": m})
    out = [o for o in out if o.get("governor") is not None]
    rng.shuffle(out)
    return out[: CAP.get(variant, DEFAULT_CAP)]


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
    "usage": (load("prolex_proposals_casual-v2.jsonl"), both_labels(pl)),
}
key = lambda d: tuple(sorted(d.items()))
for task, (source, cands) in tasks.items():
    for variant in variants:
        src = rulebook(source, variant, random.Random(5))
        cnd = [records(r, variant)[0] for r in cands]
        counts = collections.Counter(key(d) for d in src)
        count_rate = within_target(cands, [counts[key(d)] for d in cnd])[0]
        res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=src, label="rulebook"),
                                 artifact_dataset=types.SimpleNamespace(episodes=cnd, label="candidates"), max_qubits=20, seed=42)
        rate, n = within_target(cands, [None if s is None else -s for s in res["artifact_scores"]])
        seen = sum(1 for d in cnd if counts[key(d)])
        print(f"{task} / {variant}: {len(src)} rulebook records, {seen}/{len(cnd)} candidates seen; k={res['selected'].get('k')}; "
              f"FidelityAI {rate:.3f}, count {count_rate:.3f} of {n} pairs", flush=True)
