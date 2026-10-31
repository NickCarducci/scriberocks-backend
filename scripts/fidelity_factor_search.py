"""Which factors of English should a rulebook hold? Every combination of ten
closed clause factors (examples/clause_factors.rs) at k = 2-4, fit on stable
Simple English Wikipedia sentences only. Real corrections choose between factor
sets but never train a model: combinations are chosen on one sample of editors'
corrections and reported once on another. Plain counts of the same tuples get the
same search, so the comparison is fair. Score: the after's worst tuple is more
typical than the before's (0.5 chance), on edits whose tuples change, averaged
over function-word and inflection fixes. Local and free. Needs the engine's raw
role-embedding counts.

usage: OMP_NUM_THREADS=2 ~/.venvs/fidelityai/bin/python -I fidelity_factor_search.py <core dir> <grammar dir> <engine dir> <work dir> [rulebook = 5000] [workers = 4]
"""
import collections, itertools, json, multiprocessing as mp, os, random, subprocess, sys, types

core, gdir, engine, work = sys.argv[1:5]
n_rule = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
FACTORS = "SDAVOQNCMP"
DIM = {2: 16, 3: 16, 4: 8}
PER = 600


def parse(texts, name):
    with open(f"{work}/{name}.txt", "w") as f:
        f.write("\n".join(t.replace("\n", " ") for t in texts) + "\n")
    subprocess.run([f"{core}/target/release/examples/clause_factors", f"{work}/{name}.txt", f"{work}/{name}.jsonl"], cwd=core, check=True)
    return [json.loads(l) for l in open(f"{work}/{name}.jsonl")]


def load():
    rule_text = [json.loads(l)["text"] for l in open(f"{gdir}/rulebook.jsonl")]
    random.Random(3).shuffle(rule_text)
    edits = collections.defaultdict(list)
    for l in open(f"{gdir}/edits.jsonl"):
        e = json.loads(l)
        if e["kind"] != "split":
            edits[e["kind"]].append((e["before"]["text"], e["after"]["text"]))
    rng = random.Random(9)
    tune, test = [], []
    for kind, es in sorted(edits.items()):
        rng.shuffle(es)
        tune += [(kind, b, a) for b, a in es[:PER]]
        test += [(kind, b, a) for b, a in es[PER:2 * PER]]
    return rule_text[:n_rule], tune, test


def tuples(clauses, combo):
    return [[f"{f}:{c[f]}" for f in combo] for c in clauses]


def evaluate(args):
    combo, rule, sets = args
    from fidelityai.quantum import abcd_fit_and_score
    K, ids = len(combo), {}
    enc = lambda c: [ids.setdefault(v, len(ids)) for t in tuples(c, combo) for v in t]
    source = [s for s in map(enc, rule) if s]
    counts = collections.Counter(tuple(t) for c in rule for t in tuples(c, combo))
    all_cands, slices = [], {}
    for name, (pairs, sides) in sets.items():
        idx = [i for i in range(len(pairs)) if sides[2 * i] and sides[2 * i + 1]
               and tuples(sides[2 * i], combo) != tuples(sides[2 * i + 1], combo)]
        slices[name] = (len(all_cands), idx)
        all_cands += [enc(sides[j]) for i in idx for j in (2 * i, 2 * i + 1)]
    if not all_cands:
        return combo, None
    try:
        sink = open(os.devnull, "w")
        stdout, sys.stdout = sys.stdout, sink
        res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[[{"sequence": s}] for s in source], label="f"),
                                 artifact_dataset=types.SimpleNamespace(episodes=[[{"sequence": s}] for s in all_cands], label="c"),
                                 max_qubits=20, seed=42, encoding={"type": "role_embedding_v1", "dim": DIM[K], "one_hot_roles": []}, k_values=[K])
        sys.stdout = stdout
        fid = res["pipe"].tuple_scores(res["mapped_artifacts"])
        if len(fid) != len(all_cands):
            return combo, None
    except Exception as exc:
        sys.stdout = stdout
        return combo, {"error": str(exc)[:80]}
    out = {"rank": res["pipe"]._T.shape[1]}
    for name, (start, idx) in slices.items():
        pairs, sides = sets[name]
        by = collections.defaultdict(lambda: {"f": [], "c": []})
        for m, i in enumerate(idx):
            b, a = fid[start + 2 * m], fid[start + 2 * m + 1]
            cb = max(-counts[tuple(t)] for t in tuples(sides[2 * i], combo))
            ca = max(-counts[tuple(t)] for t in tuples(sides[2 * i + 1], combo))
            d = by[pairs[i][0]]
            d["f"].append((max(a) < max(b)) + 0.5 * (max(a) == max(b)))
            d["c"].append((ca < cb) + 0.5 * (ca == cb))
        for kind, d in by.items():
            out[f"{name}.{kind}"] = {"n": len(d["f"]), "f": sum(d["f"]) / len(d["f"]), "c": sum(d["c"]) / len(d["c"])}
    return combo, out


if __name__ == "__main__":
    sys.path.insert(0, engine)
    rule_text, tune, test = load()
    rule = parse(rule_text, "rule")
    sets = {"tune": (tune, parse([t for _, b, a in tune for t in (b, a)], "tune")),
            "test": (test, parse([t for _, b, a in test for t in (b, a)], "test"))}
    combos = ["".join(c) for k in (2, 3, 4) for c in itertools.combinations(FACTORS, k)]
    print(f"{len(combos)} combinations; rulebook {len(rule)}", flush=True)
    results = {}
    with mp.Pool(workers) as pool, open(f"{work}/results.jsonl", "w") as f:
        for n, (combo, r) in enumerate(pool.imap_unordered(evaluate, [(c, rule, sets) for c in combos])):
            results[combo] = r
            f.write(json.dumps({"combo": combo, "r": r}) + "\n"); f.flush()
            if (n + 1) % 50 == 0:
                print(f"done {n + 1}", flush=True)
    KINDS = ("function", "inflection")
    MIN = 100
    good = {c: r for c, r in results.items() if r and "error" not in r and all(r.get(f"tune.{k}", {}).get("n", 0) >= MIN and r.get(f"test.{k}", {}).get("n", 0) >= MIN for k in KINDS)}
    avg = lambda r, s, m: sum(r[f"{s}.{k}"][m] for k in KINDS) / 2
    print(f"{len(good)} combinations with at least {MIN} changed edits per kind in both samples; "
          f"{sum(1 for r in results.values() if r and 'error' in r)} errors")
    for label, m in (("FidelityAI", "f"), ("counts", "c")):
        print(f"best on tuning for {label}:")
        for c in sorted(good, key=lambda c: -avg(good[c], "tune", m))[:5]:
            r = good[c]
            print(f"  {c} rank {r['rank']}: tune {avg(r, 'tune', m):.3f}; TEST " + ", ".join(f"{k} F {r[f'test.{k}']['f']:.3f} C {r[f'test.{k}']['c']:.3f} (n={r[f'test.{k}']['n']})" for k in KINDS))
    wins = sum(avg(r, "test", "f") > avg(r, "test", "c") for r in good.values())
    print(f"test sample: FidelityAI above counts on {wins} of {len(good)}; correlation of tuning with test scores across combinations: ", end="")
    import numpy as np
    if len(good) > 2:
        print(round(float(np.corrcoef([avg(r, "tune", "f") for r in good.values()], [avg(r, "test", "f") for r in good.values()])[0, 1]), 3))
