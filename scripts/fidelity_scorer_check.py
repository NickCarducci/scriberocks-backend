"""Does FidelityAI's score track typicality? A check on real data, local and
free. Fit a rulebook; score three groups: seen (copies of rulebook entries),
held-out (unseen entries from the same corpus), foreign (real entries from a
different corpus). A working scorer ranks seen and held-out as more typical
than foreign. Reported per group: mean score (higher = more anomalous), share
at exactly 0 (inside the one-class boundary), and AUC (area under the ROC
curve: how often an entry of the group scores more typical than a foreign
one; 0.5 chance), for each score FidelityAI returns.

  records: SWORDS substitution records (words: target, substitute, governor,
           part of speech, relation); foreign = ProLex records
  prose:   stable Simple English Wikipedia sentences; foreign = Tatoeba

usage: ~/.venvs/fidelityai/bin/python -I fidelity_scorer_check.py <core data dir> <engine dir> [records|prose ...]
"""
import collections, json, random, sys, time, types

data, engine = sys.argv[1], sys.argv[2]
kinds = sys.argv[3:] or ["records", "prose"]
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rng = random.Random(23)


def words_record(r):
    out = {"pos": r["pos"], "rel": r["rel"], "target": r["target"], "substitute": r["substitute"], "governor": r["governor"]}
    return {k: v for k, v in out.items() if v is not None}


def auc(xs, ys):
    """How often an x (group) scores lower (more typical) than a y (foreign)."""
    return sum((x < y) + 0.5 * (x == y) for x in xs for y in ys) / max(len(xs) * len(ys), 1)


def check(name, rulebook, groups):
    cands, owner = [], []
    for g, items in groups.items():
        cands += items; owner += [g] * len(items)
    t = time.time()
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=rulebook, label="rulebook"),
                             artifact_dataset=types.SimpleNamespace(episodes=cands, label="check"), max_qubits=20, seed=42)
    gr, sel = res["pipe"].grammar, res["selected"]
    print(f"\n{name}: rulebook {len(rulebook)}; path {res['effective_path']}, encoding {(gr.encoding or {}).get('type', 'exact index')}, "
          f"k={sel.get('k')}, v={sel.get('v')}, qubits={sel.get('n_qubits')}, columns {(gr.columns or {}).get('kept')}, {time.time() - t:.0f} s")
    prov = res.get("artifact_provenance") or []
    # The signed one-class score (higher = more anomalous), before any distance.
    pipe = res["pipe"]
    mapped = res.get("mapped_artifacts") or []
    res["signed_boundary"] = pipe._ocsvm_boundary_score(pipe._project_features(mapped)).tolist() if mapped else []
    # How much of each record lies outside what the reference bank spans.
    if mapped:
        from fidelityai.fitter import extract_batch
        feats, _ = extract_batch(mapped, pipe.grammar, pipe._active_dictionary_profile(), fit_scaler=False, scaler=pipe._scaler)
        res["residual"] = pipe._reconstruction_residual(feats).tolist()
    for key in ("artifact_scores", "signed_boundary", "residual", "hier_artifact_scores", "hier_direct_artifact_scores"):
        scores = res.get(key) or []
        # Map scores to candidates: per sequence (provenance sourceIndex) or one per candidate.
        if key in ("artifact_scores", "signed_boundary", "residual") and prov and len(prov) == len(scores) and isinstance(prov[0], dict) and "sourceIndex" in prov[0]:
            per = collections.defaultdict(list)
            for s, p in zip(scores, prov):
                per[p["sourceIndex"]].append(s)
            by_cand = {i: sum(v) / len(v) for i, v in per.items()}
        elif len(scores) == len(cands):
            by_cand = dict(enumerate(scores))
        elif key != "artifact_scores" and prov:
            idx = sorted({p["sourceIndex"] for p in prov if isinstance(p, dict) and "sourceIndex" in p})
            by_cand = dict(zip(idx, scores)) if len(idx) == len(scores) else {}
        else:
            by_cand = {}
        if not by_cand:
            continue
        g = collections.defaultdict(list)
        for i, s in by_cand.items():
            if s is not None:
                g[owner[i]].append(s)
        line = []
        for name_g in groups:
            xs = g[name_g]
            if not xs:
                continue
            stats = f"{name_g} mean {sum(xs) / len(xs):.4f}, zero {100 * sum(x == 0 for x in xs) / len(xs):.0f}%"
            if name_g != "foreign":
                stats += f", AUC vs foreign {auc(xs, g['foreign']):.3f}"
            line.append(stats)
        print(f"  {key} ({len(set(round(v, 9) for v in by_cand.values()))} distinct): " + "; ".join(line))


if "records" in kinds:
    sw = [json.loads(l) for l in open(f"{data}/fidelity-edits/swords_subs.jsonl")]
    pl = [json.loads(l) for l in open(f"{data}/fidelity-edits/prolex_subs.jsonl")]
    acc = [words_record(r) for r in sw if r["y"] == 1]
    rng.shuffle(acc)
    rulebook, held = acc[:2000], acc[2000:2600]
    check("records (SWORDS accepted substitutions)", rulebook,
          {"seen": rng.sample(rulebook, 600), "held-out": held, "foreign": [words_record(r) for r in rng.sample(pl, 600)]})

if "prose" in kinds:
    stable = [l.rstrip("\n").split("\t")[3] for l in open(f"{data}/simplewiki-history/stable_train.tsv")]
    rng.shuffle(stable)
    tatoeba = [l.strip() for l in open(f"{data}/corpus/tatoeba.txt") if 6 <= len(l.split()) <= 30]
    rulebook, held = stable[:2000], stable[2000:2300]
    check("prose (stable Simple Wikipedia sentences)", rulebook,
          {"seen": rng.sample(rulebook, 300), "held-out": held, "foreign": rng.sample(tatoeba, 300)})
