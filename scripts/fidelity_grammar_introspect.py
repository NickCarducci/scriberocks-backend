"""FidelityAI's element introspection on editors' grammar corrections: not
the whole record's score, but each tuple's contribution to it (scorer.js
tupleContributions: margin sensitivity through the tuple's features, the
Frobenius spider's per-element reading), plus whether the tuple is in the
rulebook. Same data as fidelity_grammar_eval.py: Simple English Wikipedia
corrections, rulebook of 5,000 stable sentences, sentences as fine part-of-
speech tag sequences (path D). Local and free.

  detect  the after scores more typical than the before when a sentence is
          scored by its most frustrated tuple, or by its count of tuples the
          rulebook never had (0.5 chance)
  locate  in the before (the erroneous text), the most frustrated tuple
          covers the edited word (chance: the share of tuples that cover it)

usage: ~/.venvs/fidelityai/bin/python -I fidelity_grammar_introspect.py <grammar dir> <engine dir> [per kind = 600]
"""
import collections, json, random, subprocess, sys, types
from pathlib import Path

gdir, engine = sys.argv[1], sys.argv[2]
per_kind = int(sys.argv[3]) if len(sys.argv) > 3 else 600
sys.path.insert(0, engine)
from fidelityai.quantum import _pipe_to_score_artifact, abcd_fit_and_score  # noqa: E402

rulebook = [json.loads(l) for l in open(f"{gdir}/rulebook.jsonl")]
edits = collections.defaultdict(list)
for l in open(f"{gdir}/edits.jsonl"):
    e = json.loads(l)
    edits[e["kind"]].append(e)
rng = random.Random(9)
pairs = []
for kind, es in sorted(edits.items()):
    rng.shuffle(es)
    pairs += es[:per_kind]

tag_ids = {}
tid = lambda t: tag_ids.setdefault(t, len(tag_ids))
flat = lambda d, key: [x for s in d["sentences"] for x in ["<s>"] + s[key]]
seq = lambda d: [tid(t) for t in flat(d, "xpos")]
for d in rulebook + [p[s] for p in pairs for s in ("before", "after")]:
    seq(d)


def edit_index(p):
    """First position (in the before's flattened tokens) where the edit differs."""
    b, a = [w.lower() for w in flat(p["before"], "tokens")], [w.lower() for w in flat(p["after"], "tokens")]
    i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return min(i, len(b) - 1)


texts = [p[s] for p in pairs for s in ("before", "after")]
res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[[{"sequence": seq(d)}] for d in rulebook], label="stable-tags"),
                         artifact_dataset=types.SimpleNamespace(episodes=[[{"sequence": seq(d)}] for d in texts], label="candidates"),
                         max_qubits=20, seed=42)
pipe, mapped = res["pipe"], res["mapped_artifacts"]
k = int(pipe.grammar.tuple_size)
print(f"fit: path {res['effective_path']}, encoding {(pipe.grammar.encoding or {}).get('type', 'exact index')}, k={k}; "
      f"{len(mapped)} mapped of {len(texts)} texts", flush=True)
assert len(mapped) == len(texts)

cli = Path(engine) / "scripts" / "scorer_cli.mjs"
payload = {"artifact": _pipe_to_score_artifact(pipe), "sequences": mapped, "options": {"topContributions": 10_000}}
out = subprocess.run(["node", str(cli)], input=json.dumps(payload), capture_output=True, text=True, check=True)
rows = json.loads(out.stdout)["rows"]

detect = collections.defaultdict(lambda: {"max": [], "unseen": [], "whole": []})
locate = collections.defaultdict(lambda: [0, 0.0, 0])  # hits, chance, n
for i, p in enumerate(pairs):
    rb, ra = rows[2 * i], rows[2 * i + 1]
    cb, ca = rb.get("tupleContributions") or [], ra.get("tupleContributions") or []
    if not cb or not ca:
        continue
    worst = lambda c: max(x["contribution"] for x in c)
    unseen = lambda c: sum(x["count"] for x in c if not x["inRulebook"])
    d = detect[p["kind"]]
    d["max"].append((worst(ca) < worst(cb)) + 0.5 * (worst(ca) == worst(cb)))
    d["unseen"].append((unseen(ca) < unseen(cb)) + 0.5 * (unseen(ca) == unseen(cb)))
    d["whole"].append((ra["score"] < rb["score"]) + 0.5 * (ra["score"] == rb["score"]))
    # Locate: the before's most frustrated tuple covers the edited position.
    e = edit_index(p)
    n_grams = max(len(mapped[2 * i]) - k + 1, 1)
    covering = [pos for pos in range(n_grams) if pos <= e <= pos + k - 1]
    top = max(cb, key=lambda x: x["contribution"])
    L = locate[p["kind"]]
    L[0] += any(pos in covering for pos in top["positions"])
    L[1] += min(1.0, len(covering) * len(top["positions"]) / n_grams)
    L[2] += 1

for kind in sorted(detect):
    d, L = detect[kind], locate[kind]
    m = lambda xs: sum(xs) / max(len(xs), 1)
    print(f"{kind} ({len(d['max'])} edits): detect — whole record {m(d['whole']):.3f}, most frustrated tuple {m(d['max']):.3f}, "
          f"tuples outside the rulebook {m(d['unseen']):.3f}; locate — top tuple covers the edit {L[0] / max(L[2], 1):.3f} "
          f"(chance {L[1] / max(L[2], 1):.3f})")
