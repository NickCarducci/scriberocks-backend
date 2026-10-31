"""Confirm factor-search winners (fidelity_factor_search.py) on a third sample of
editors' corrections, untouched by tuning or the search's report. Writes each
combination's changed edits for the judge. CUTOFF=<0-1> sets how much of the
reference overlaps the projection keeps (engine default 0.9), in this process only.

usage: [CUTOFF=0.99] ~/.venvs/fidelityai/bin/python -I fidelity_factor_confirm.py <core dir> <grammar dir> <engine dir> <work dir> <rulebook> <combo,combo,...>
"""
import importlib.util, os, random, sys, collections, json
core, gdir, engine, work, n_rule, combos = sys.argv[1:7]
sys.path.insert(0, engine)
spec = importlib.util.spec_from_file_location("fs", f"{core}/scripts/fidelity_factor_search.py")
argv, sys.argv = sys.argv, ["x", core, gdir, engine, work, n_rule]
fs = importlib.util.module_from_spec(spec); spec.loader.exec_module(fs); sys.argv = argv
cut = float(os.environ.get("CUTOFF", 0.9))
if cut != 0.9:
    import numpy as _np, fidelityai.fitter as ft
    class P:
        def __getattr__(self, a): return getattr(_np, a)
        @staticmethod
        def searchsorted(a, v, *r, **k): return _np.searchsorted(a, cut if v == 0.9 else v, *r, **k)
    ft.np = P()
rule_text = [json.loads(l)["text"] for l in open(f"{gdir}/rulebook.jsonl")]
random.Random(3).shuffle(rule_text)
edits = collections.defaultdict(list)
for l in open(f"{gdir}/edits.jsonl"):
    e = json.loads(l)
    if e["kind"] != "split":
        edits[e["kind"]].append((e["before"]["text"], e["after"]["text"]))
rng = random.Random(9)
third = []
for kind, es in sorted(edits.items()):
    rng.shuffle(es)
    third += [(kind, b, a) for b, a in es[2 * fs.PER:3 * fs.PER]]
rule = fs.parse(rule_text[:int(n_rule)], "c_rule")
sides = fs.parse([t for _, b, a in third for t in (b, a)], "c_third")
sets = {"third": (third, sides)}
for combo in combos.split(","):
    _, r = fs.evaluate((combo, rule, sets))
    print(f"{combo} rulebook {n_rule} cutoff {cut} rank {r['rank']}: " + ", ".join(
        f"{k} F {r[f'third.{k}']['f']:.3f} C {r[f'third.{k}']['c']:.3f} (n={r[f'third.{k}']['n']})" for k in ("function", "inflection")), flush=True)
    with open(f"{work}/judge_{combo}.tsv", "w") as f:
        for i, (k, b, a) in enumerate(third):
            if sides[2 * i] and sides[2 * i + 1] and fs.tuples(sides[2 * i], combo) != fs.tuples(sides[2 * i + 1], combo):
                f.write(f"{k}\t{i}\t0\t{b}\n{k}\t{i}\t1\t{a}\n")
