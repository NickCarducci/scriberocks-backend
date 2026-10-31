"""Preprocessing finds the pattern, FidelityAI judges how it combines: tuples of
words the parser links at any distance (examples/linked_tuples.rs), closed
features only, on editors' grammar corrections. Rulebook: stable Simple English
Wikipedia sentences. Local and free.

Per edit: does the after's worst tuple score more typical than the before's
(0.5 chance), over all edits and over those whose tuples actually change? Does
the before's worst tuple read the edited word? Baseline: plain counts of the same
tuples in the rulebook (rarest tuple is the worst). Also reported on the edits where those
counts tie, the sparse case an overlap could still judge. Also reports model size and
per-sentence scoring time. REFS=<n> sets the reference bank (default 30). Writes changed pairs for the judge (text_judge_eval).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_linked_eval.py <core dir> <grammar dir> <engine dir> <work dir> [rulebook = 20000] [per kind = 600]
"""
import collections, json, os, random, subprocess, sys, time, types

core, gdir, engine, work = sys.argv[1:5]
n_rule = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
per_kind = int(sys.argv[6]) if len(sys.argv) > 6 else 600
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402


def parse(texts, name):
    with open(f"{work}/{name}.txt", "w") as f:
        f.write("\n".join(t.replace("\n", " ") for t in texts) + "\n")
    subprocess.run([f"{core}/target/release/examples/linked_tuples", f"{work}/{name}.txt", f"{work}/{name}.jsonl"], cwd=core, check=True)
    return [json.loads(l) for l in open(f"{work}/{name}.jsonl")]


rule_text = [json.loads(l)["text"] for l in open(f"{gdir}/rulebook.jsonl")]
random.Random(3).shuffle(rule_text)
edits = collections.defaultdict(list)
for l in open(f"{gdir}/edits.jsonl"):
    e = json.loads(l)
    if e["kind"] != "split":
        edits[e["kind"]].append((e["before"]["text"], e["after"]["text"]))
rng = random.Random(9)
pairs = []
for kind, es in sorted(edits.items()):
    rng.shuffle(es)
    pairs += [(kind, b, a) for b, a in es[:per_kind]]
rule = parse(rule_text[:n_rule], "rule")
sides = parse([t for _, b, a in pairs for t in (b, a)], "cands")
print(f"rulebook {len(rule)} sentences; " + ", ".join(f"{k} {sum(p[0] == k for p in pairs)}" for k in sorted(edits)), flush=True)

VARIANTS = {"pairs": (("pairs",), 2, 16), "clause": (("clause",), 4, 8), "np": (("np",), 4, 8), "clause+np": (("clause", "np"), 4, 8)}


def edit_at(b, a):
    i = next((i for i in range(min(len(a), len(b))) if a[i].lower() != b[i].lower()), min(len(a), len(b)))
    return min(i, max(len(b) - 1, 0))


for name, (fields, K, dim) in VARIANTS.items():
    tuples = lambda d: [t for f in fields for t in d[f]]
    ids = {}
    enc = lambda t: [ids.setdefault(v, len(ids)) for v in t["v"]]
    seq = lambda d: [x for t in tuples(d) for x in enc(t)]
    keep = [i for i in range(len(pairs)) if tuples(sides[2 * i]) and tuples(sides[2 * i + 1])]
    source = [s for s in map(seq, rule) if s]
    cands = [seq(sides[j]) for i in keep for j in (2 * i, 2 * i + 1)]
    counts = collections.Counter(tuple(t["v"]) for d in rule for t in tuples(d))
    t0 = time.time()
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[[{"sequence": s}] for s in source], label="linked"),
                             artifact_dataset=types.SimpleNamespace(episodes=[[{"sequence": s}] for s in cands], label="candidates"),
                             max_qubits=20, seed=42, encoding={"type": "role_embedding_v1", "dim": dim, "one_hot_roles": []}, k_values=[K],
                             max_benign_refs=int(os.environ.get("REFS", 30)))
    pipe, fit_s = res["pipe"], time.time() - t0
    assert len(res["mapped_artifacts"]) == len(cands), (len(res["mapped_artifacts"]), len(cands))
    t1 = time.time()
    fid = pipe.tuple_scores(res["mapped_artifacts"])
    ms = 1000 * (time.time() - t1) / max(len(cands), 1)
    stat = collections.defaultdict(lambda: collections.defaultdict(list))
    for m, i in enumerate(keep):
        kind = pairs[i][0]
        db, da = sides[2 * i], sides[2 * i + 1]
        tb, ta = tuples(db), tuples(da)
        changed = [t["v"] for t in tb] != [t["v"] for t in ta]
        e = edit_at(db["tokens"], da["tokens"])
        for method, b, a in (("FidelityAI", fid[2 * m], fid[2 * m + 1]),
                             ("counts", [-counts[tuple(t["v"])] for t in tb], [-counts[tuple(t["v"])] for t in ta])):
            win = (max(a) < max(b)) + 0.5 * (max(a) == max(b))
            s = stat[kind]
            s[method + " all"].append(win)
            if changed:
                s[method + " changed"].append(win)
                if method == "FidelityAI":
                    cb, ca = [-counts[tuple(t["v"])] for t in tb], [-counts[tuple(t["v"])] for t in ta]
                    # Where counts can't help: their worst tuples tie.
                    if max(cb) == max(ca):
                        s["FidelityAI tied"].append(win)
                    # Where the rulebook never saw a worst tuple on either side.
                    if max(cb) == 0 and max(ca) == 0:
                        s["FidelityAI unseen"].append(win)
                worst = [j for j, v in enumerate(b) if v == max(b)]
                s[method + " locate"].append(sum(e in tb[j]["at"] for j in worst) / len(worst))
                if method == "FidelityAI":
                    s["chance"].append(sum(e in t["at"] for t in tb) / len(tb))
    avg = lambda x: sum(x) / max(len(x), 1)
    T = pipe._T
    print(f"{name}: k={K}, dim={dim}, qubits={res['selected'].get('n_qubits')}, v={len(ids)}, projection {T.shape[0]}x{T.shape[1]} ({T.size * 8 / 1e6:.1f} MB raw), "
          f"fit {fit_s:.0f} s, scoring {ms:.2f} ms per text", flush=True)
    for kind in sorted(stat):
        s = stat[kind]
        print(f"  {kind}: changed {len(s['FidelityAI changed'])}/{len(s['FidelityAI all'])}; "
              f"detect changed FidelityAI {avg(s['FidelityAI changed']):.3f}, counts {avg(s['counts changed']):.3f}; "
              f"all edits FidelityAI {avg(s['FidelityAI all']):.3f}, counts {avg(s['counts all']):.3f}; "
              f"locate FidelityAI {avg(s['FidelityAI locate']):.3f}, counts {avg(s['counts locate']):.3f}, chance {avg(s['chance']):.3f}", flush=True)
        print(f"    where counts tie: FidelityAI {avg(s['FidelityAI tied']):.3f} ({len(s['FidelityAI tied'])}); "
              f"both sides hold an unseen tuple: FidelityAI {avg(s['FidelityAI unseen']):.3f} ({len(s['FidelityAI unseen'])})", flush=True)
    if name == "clause+np":
        with open(f"{work}/judge_changed.tsv", "w") as f:
            for i in keep:
                kind, b, a = pairs[i]
                if [t["v"] for t in tuples(sides[2 * i])] != [t["v"] for t in tuples(sides[2 * i + 1])]:
                    f.write(f"{kind}\t{i}\t0\t{b}\n{kind}\t{i}\t1\t{a}\n")
