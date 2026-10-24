"""FidelityAI on whole rocks: one document per rock (the text after editing),
so the cascade (path B: grammar frames, intra/inter-tuple and congruency
structure) learns what a good rock looks like. Free and local.

Source: edited ("after") sentences from Simple English Wikipedia training
articles. Test: held-out edits, both versions scored as documents; reported
is how often the after version scores as more typical than the before one.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_rock_docs.py <simplewiki-history dir> <engine dir> <out tsv> [source n = 5000] [test n = 1000]
"""
import collections, sys, time, types

hist, engine, out = sys.argv[1], sys.argv[2], sys.argv[3]
n_src = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
n_test = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402


def read(path, n):
    rows = []
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3 and p[1] != p[2]:
            rows.append((p[1], p[2]))
        if len(rows) >= n:
            break
    return rows


source = [after for _, after in read(f"{hist}/train.tsv", n_src)]
test = read(f"{hist}/test.tsv", n_test)
docs = [t for pair in test for t in pair]  # before, after, before, after, ...

t = time.time()
res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=source, label="rocks"),
                         artifact_dataset=types.SimpleNamespace(episodes=docs, label="candidates"), max_qubits=20, seed=42)
sel = res["selected"]
print(f"path {res['effective_path']}, k={sel.get('k')}, v={sel.get('v')}, qubits={sel.get('n_qubits')}, {time.time() - t:.0f} s")

# Per document: frame scores (distance from the fit; larger = less typical) by provenance.
per = collections.defaultdict(list)
for sc, pv in zip(res["artifact_scores"], res.get("artifact_provenance") or []):
    if sc is not None and isinstance(pv, dict) and isinstance(pv.get("sourceIndex"), int):
        per[pv["sourceIndex"]].append(sc)
hier = res.get("hier_artifact_scores") or []
print(f"{len(per)} of {len(docs)} documents have frame scores; {len(hier)} episode-level scores")

with open(out, "w") as f:
    for i in range(0, len(docs), 2):
        b, a = per.get(i), per.get(i + 1)
        hb = hier[i] if len(hier) == len(docs) else ""
        ha = hier[i + 1] if len(hier) == len(docs) else ""
        f.write(f"{sum(b) / len(b) if b else ''}\t{sum(a) / len(a) if a else ''}\t{hb}\t{ha}\t{docs[i]}\t{docs[i + 1]}\n")


def better(pairs):
    pairs = [(b, a) for b, a in pairs if b != "" and a != ""]
    wins = sum((a < b) + 0.5 * (a == b) for b, a in pairs)
    return wins / max(len(pairs), 1), len(pairs)


frame = better([(sum(per[i]) / len(per[i]) if per.get(i) else "", sum(per[i + 1]) / len(per[i + 1]) if per.get(i + 1) else "") for i in range(0, len(docs), 2)])
print(f"after scored more typical than before (frame mean): {frame[0]:.3f} of {frame[1]} edits")
if len(hier) == len(docs):
    h = better([(hier[i], hier[i + 1]) for i in range(0, len(docs), 2)])
    print(f"after scored more typical than before (document level): {h[0]:.3f} of {h[1]} edits")
