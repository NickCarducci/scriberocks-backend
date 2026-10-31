"""Misnomers: dictionary synonyms that mean the same but don't pair the same way
(strong tea, powerful tea). Only targets where accepted and rejected substitutes are
both WordNet synonyms of the original word, so every candidate passes on meaning. A
rulebook of real word pairings with co-occurrence embeddings judges each candidate's
overlapping pairs, (governor, relation, word) and (word, relation, dependent).
Fit on real text only; ProLex stays local. Within target (0.5 chance), on all pairs and
on pairs of similar word frequency. Writes the pairs for the judge.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_misnomer_eval.py <colloc prefix> <subs.tsv> <subs records.jsonl> <wordnet.jsonl> <engine dir> <out prefix> <sizes>
"""
import collections, json, math, random, sys

prefix, tsv, records, wordnet, engine, out, sizes = sys.argv[1:8]
sys.path.insert(0, engine)
from fidelityai.fitter import HandoffPipeline  # noqa: E402
from fidelityai.grammar import GrammarSpec  # noqa: E402

DIM, K = 16, 3
synsets = collections.defaultdict(set)
for l in open(wordnet):
    s = json.loads(l)
    for m in s["members"]:
        synsets[m.lower()].add(s["id"])
target = {}
for l in open(records):
    r = json.loads(l)
    target[r["tid"]] = r["target"].lower()
rows = [l.rstrip("\n").split("\t") for l in open(tsv)]
cands = [json.loads(l) for l in open(prefix + ".candidates.jsonl")]
# Align candidates with their text: collocations skips rows it can't locate.
text, j = [], 0
for c in cands:
    while not (rows[j][1] == c["tid"] and int(rows[j][2]) == c["y"]):
        j += 1
    r = rows[j]; text.append(" ".join(f"{r[3]} {r[5]} {r[6]}".split())); j += 1
word = lambda c: c["up"]["dep"] if c["up"] else (c["down"][0]["gov"] if c["down"] else None)
freq = lambda c: c["up"]["dep_n"] if c["up"] else (c["down"][0]["gov_n"] if c["down"] else 0)
triples = lambda c: ([c["up"]] if c["up"] else []) + c["down"]
syn = lambda c: word(c) and synsets[word(c).lower()] & synsets[target.get(c["tid"], "")]
groups = collections.defaultdict(lambda: ([], []))
for i, c in enumerate(cands):
    if triples(c) and syn(c):
        groups[c["tid"]][1 - c["y"]].append(i)
groups = {t: g for t, g in groups.items() if g[0] and g[1]}
pairs = [(a, b) for acc, rej in groups.values() for a in acc for b in rej]
close = [p for p in pairs if max(freq(cands[p[0]]), freq(cands[p[1]])) <= 2 * max(min(freq(cands[p[0]]), freq(cands[p[1]])), 1)]
print(f"{len(groups)} targets, {len(pairs)} pairs of synonyms, {len(close)} of similar frequency", flush=True)
with open(out + ".all.tsv", "w") as f:
    for n, (a, b) in enumerate(pairs):
        f.write(f"{n}\t1\t{text[a]}\n{n}\t0\t{text[b]}\n")
with open(out + ".close.tsv", "w") as f:
    for n, (a, b) in enumerate(close):
        f.write(f"{n}\t1\t{text[a]}\n{n}\t0\t{text[b]}\n")
used = sorted({i for p in pairs for i in p})
tk = lambda t: (t["gov"], t["rel"], t["dep"])
ups = {(t["gov"], t["rel"]) for i in used for t in triples(cands[i])}
downs = {(t["rel"], t["dep"]) for i in used for t in triples(cands[i])}
keys = collections.defaultdict(list)
for l in open(prefix + ".rulebook.tsv"):
    g, r, d, n = l.rstrip("\n").split("\t")
    if (g, r) in ups: keys[("u", g, r)].append(((g, r, d), int(n)))
    if (r, d) in downs: keys[("d", r, d)].append(((g, r, d), int(n)))
ks = sorted(keys)


def rate(s, ps):
    return sum((s[a] > s[b]) + 0.5 * (s[a] == s[b]) for a, b in ps) / max(len(ps), 1)


for n_rule in map(int, sizes.split(",")):
    rng = random.Random(13)
    per = max(1, n_rule // max(len(ks), 1))
    rule = []
    for k in ks:
        ts, ws = zip(*keys[k])
        rule += rng.choices(ts, weights=ws, k=per)
    rng.shuffle(rule); rule = rule[:n_rule]
    counts = collections.Counter(rule)
    ids = {}
    tid = lambda p, v: ids.setdefault((p, v), len(ids))
    seq = lambda t: [tid(p, v) for p, v in enumerate(t)]
    source = [seq(t) for t in rule]
    flat, owner = [], []
    for i in used:
        for t in triples(cands[i]):
            flat.append(seq(tk(t))); owner.append(i)
    g = GrammarSpec(roles=[f"t{i}" for i in range(len(ids))], allowed_transitions={i: [] for i in range(len(ids))}, forbidden_tuples=set(), benign_tuples=set(),
                    tuple_size=K, n_qubits=K * 4 + 2, label="pairs", encoding={"type": "role_embedding_v1", "dim": DIM, "one_hot_roles": []})
    pipe = HandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42).fit(source)
    sc = pipe.score(flat)
    by = collections.defaultdict(list)
    for i, v in zip(owner, sc):
        by[i].append(-float(v))
    fm = {i: sum(v) / len(v) for i, v in by.items()}
    fw = {i: min(v) for i, v in by.items()}
    cnt = {i: sum(math.log1p(counts[tk(t)]) for t in triples(cands[i])) for i in used}
    frq = {i: math.log1p(freq(cands[i])) for i in used}
    print(f"rulebook {len(rule)}: " + "; ".join(
        f"{label} ({len(ps)}) mean tuple {rate(fm, ps):.3f}, worst tuple {rate(fw, ps):.3f}, counts {rate(cnt, ps):.3f}, frequency {rate(frq, ps):.3f}"
        for label, ps in (("all", pairs), ("similar frequency", close))), flush=True)
