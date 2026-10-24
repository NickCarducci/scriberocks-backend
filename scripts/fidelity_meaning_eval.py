"""A rulebook of known meaning sequences: does FidelityAI tell a fitting
substitute from a synonym that doesn't fit, when frame slots carry WordNet
meanings instead of words? Real data only, local and free.

Rulebook (source): clause frames of SemCor and the WordNet Gloss Corpus
(examples/meaning_frames.rs), one record per frame. Candidates: the SWORDS
sentences (accepted and rejected substitutes, from fidelity_swords_eval.py).
Slot values, per variant:
  lemma     the slot's lemma on both sides (the word-level baseline)
  gold      rulebook: annotators' synset; candidates: the sense head's
  pred      the sense head's synset on both sides
  category  as pred, coarsened to the lexicographer category (noun.artifact)
A slot with no meaning (pronouns, untagged words) keeps its lemma.
Reported: within each target, how often an accepted version scores more
typical than a rejected one (0.5 chance), over targets whose candidates'
records differ (the others are ties by construction).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_meaning_eval.py <meaning_frames.jsonl> <engine dir> [variants] [rulebook records = 6000]
"""
import collections, json, random, sys, time, types

path, engine = sys.argv[1], sys.argv[2]
variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["lemma", "gold", "pred", "category"]
n_source = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rows = [json.loads(l) for l in open(path)]
annotated = [r for r in rows if r["corpus"] != "swords"]
swords = [r for r in rows if r["corpus"] == "swords"]
SLOTS = ("subject", "action", "object", "target")


def value(slot, side, variant):
    if not slot:
        return None
    if variant == "lemma":
        return slot["lemma"]
    key = "gold" if variant == "gold" and side == "source" else "pred"
    m = slot.get(key)
    if not m:
        return slot["lemma"]
    return m["c"] if variant == "category" else m["s"]


def record(frame, side, variant):
    r = {s: value(frame.get(s), side, variant) for s in SLOTS}
    r["modal"] = frame.get("modal")
    r["status"] = frame["status"]
    return {k: v for k, v in r.items() if v is not None}


def within_target(sentence_scores, keep):
    groups = collections.defaultdict(lambda: ([], []))
    for r, s in zip(swords, sentence_scores):
        if s is not None and r["tid"] in keep:
            groups[r["tid"]][1 - r["y"]].append(s)
    wins = n = 0
    for acc, rej in groups.values():
        for a in acc:
            for b in rej:
                wins += (a > b) + 0.5 * (a == b); n += 1
    return wins / max(n, 1), n


rng = random.Random(7)
source_frames = [f for r in annotated for f in r["frames"]]
rng.shuffle(source_frames)
source_frames = source_frames[:n_source]
print(f"rulebook: {len(source_frames)} frames from {len(annotated)} annotated sentences; "
      f"candidates: {len(swords)} SWORDS sentences ({sum(r['y'] for r in swords)} accepted)")

for variant in variants:
    source = [record(f, "source", variant) for f in source_frames]
    cands, owner = [], []
    for i, r in enumerate(swords):
        for f in r["frames"]:
            cands.append(record(f, "candidate", variant)); owner.append(i)
    # Targets whose candidates' records differ.
    sig = collections.defaultdict(set)
    for i, r in enumerate(swords):
        sig[r["tid"]].add(json.dumps([record(f, "candidate", variant) for f in r["frames"]], sort_keys=True))
    keep = {t for t, s in sig.items() if len(s) > 1}
    t = time.time()
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=source, label="meanings"),
                             artifact_dataset=types.SimpleNamespace(episodes=cands, label="swords"), max_qubits=20, seed=42)
    sel, g = res["selected"], res["pipe"].grammar
    scores = res["artifact_scores"]
    if len(scores) != len(cands):
        print(f"{variant}: {len(scores)} scores for {len(cands)} records; skipped")
        continue
    per = collections.defaultdict(list)
    for i, s in zip(owner, scores):
        if s is not None:
            per[i].append(s)
    sentence = [(-sum(per[i]) / len(per[i]) if per.get(i) else None) for i in range(len(swords))]
    rate, n = within_target(sentence, keep)
    print(f"{variant}: path {res['effective_path']}, encoding {(getattr(g, 'encoding', None) or {}).get('type', 'exact index')}, "
          f"k={sel.get('k')}, qubits={sel.get('n_qubits')}, {time.time() - t:.0f} s; "
          f"{len(keep)} targets where records differ; accepted beats rejected {rate:.3f} of {n} pairs")
    with open(path.replace(".jsonl", f".{variant}.keep"), "w") as f:
        f.write("\n".join(sorted(keep)))
