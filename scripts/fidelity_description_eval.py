"""Roles known by description alone: can never-seen combinations be judged?
A value's embedding is built from its published description (Penn Treebank tag
definitions; standard English function-word features), not learned from
co-occurrence, so an unseen tuple still overlaps with seen ones through shared
attributes. Clause factors from examples/clause_factors.rs; fit on stable Simple
English Wikipedia sentences only. Local and free.

Scored only on editors' corrections where both sides hold a tuple that never appears
in the rulebook, so counts of the worst tuple tie, from a sample untouched by earlier tuning. The tables and factor
sets were fixed before any scores existed.

  description  FidelityAI with the tables below
  learned      FidelityAI with embeddings learned from the same rulebook
  attributes   classical: mean smoothed PMI of attribute pairs across positions
  counts       plain tuple counts
Reported: the after's worst tuple is more typical (0.5 chance); the before's worst
tuple is one the edit changes (with chance). be/have swaps are reported apart.

usage: ~/.venvs/fidelityai/bin/python -I fidelity_description_eval.py <core dir> <grammar dir> <engine dir> <work dir>
"""
import collections, itertools, json, math, random, re, subprocess, sys

import numpy as np

core, gdir, engine, work = sys.argv[1:5]
sys.path.insert(0, engine)
from fidelityai.fitter import HandoffPipeline  # noqa: E402
from fidelityai.grammar import GrammarSpec  # noqa: E402

COMBOS = ["SA", "SAV", "AVO", "SAVO", "DS", "QO"]
DIM = 16
OFFSET, PER = 1800, 3000

# Penn Treebank tag definitions.
TAG = {"NN": "NOUN SG", "NNS": "NOUN PL", "NNP": "NOUN SG PROP", "NNPS": "NOUN PL PROP",
       "PRP": "PRON", "PRP$": "PRON FUNC", "WP": "PRON FUNC", "WP$": "PRON FUNC", "EX": "PRON FUNC", "DT": "FUNC", "WDT": "FUNC", "PDT": "FUNC",
       "VB": "VERB BASE", "VBP": "VERB PRES", "VBZ": "VERB PRES SG THIRD", "VBD": "VERB PAST", "VBN": "VERB PPART", "VBG": "VERB GER", "MD": "VERB FUNC",
       "JJ": "MOD", "JJR": "MOD", "JJS": "MOD", "RB": "MOD", "RBR": "MOD", "RBS": "MOD", "WRB": "MOD FUNC", "CD": "MOD NOUN",
       "IN": "FUNC", "TO": "FUNC", "CC": "FUNC", "RP": "FUNC", "UH": "FUNC", "FW": "NOUN", "SYM": "FUNC", "POS": "FUNC", "LS": "FUNC",
       "-": "MISSING"}
TAG_ATTRS = "NOUN VERB PRON MOD FUNC MISSING SG PL PROP PAST PRES PPART GER BASE THIRD".split()
# Standard English auxiliary features.
AUX = {"am": "BE PRES SG FIRST", "'m": "BE PRES SG FIRST", "is": "BE PRES SG THIRD", "'s": "BE PRES SG THIRD", "are": "BE PRES PL", "'re": "BE PRES PL",
       "was": "BE PAST SG", "were": "BE PAST PL", "be": "BE BASE", "been": "BE PPART", "being": "BE GER",
       "has": "HAVE PRES SG THIRD", "have": "HAVE PRES PL", "'ve": "HAVE PRES PL", "had": "HAVE PAST", "'d": "HAVE PAST MODAL", "having": "HAVE GER",
       "do": "DO PRES PL", "does": "DO PRES SG THIRD", "did": "DO PAST", "doing": "DO GER", "done": "DO PPART",
       "will": "MODAL PRES", "'ll": "MODAL PRES", "wo": "MODAL PRES", "would": "MODAL PAST", "can": "MODAL PRES", "ca": "MODAL PRES", "could": "MODAL PAST",
       "shall": "MODAL PRES", "should": "MODAL PAST", "may": "MODAL PRES", "might": "MODAL PAST", "must": "MODAL PRES", "ought": "MODAL PRES", "to": "BASE INF", "-": "MISSING"}
AUX_ATTRS = "BE HAVE DO MODAL MISSING PRES PAST SG PL FIRST THIRD PPART GER BASE INF".split()
# Standard English determiner features.
DET = {"the": "DEF", "a": "INDEF SG", "an": "INDEF SG", "this": "DEF DEM SG", "that": "DEF DEM SG", "these": "DEF DEM PL", "those": "DEF DEM PL",
       "my": "DEF POSS", "your": "DEF POSS", "his": "DEF POSS", "her": "DEF POSS", "its": "DEF POSS", "our": "DEF POSS", "their": "DEF POSS",
       "some": "QUANT", "any": "QUANT", "no": "QUANT NEG", "each": "QUANT SG", "every": "QUANT SG", "either": "QUANT SG", "neither": "QUANT SG NEG",
       "both": "QUANT PL", "all": "QUANT", "many": "QUANT PL", "few": "QUANT PL", "several": "QUANT PL", "much": "QUANT MASS", "little": "QUANT MASS", "another": "INDEF SG", "-": "MISSING"}
DET_ATTRS = "DEF INDEF DEM POSS QUANT NEG MISSING SG PL MASS".split()
TABLE = {"S": (TAG, TAG_ATTRS), "V": (TAG, TAG_ATTRS), "O": (TAG, TAG_ATTRS), "A": (AUX, AUX_ATTRS), "D": (DET, DET_ATTRS), "Q": (DET, DET_ATTRS)}


def attrs(factor, value):
    table, _ = TABLE[factor]
    return table.get(value, "").split()


def vector(factor, value):
    _, names = TABLE[factor]
    v = np.zeros(DIM)
    for a in attrs(factor, value):
        v[names.index(a)] = 1.0
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else None


def parse(texts, name):
    with open(f"{work}/{name}.txt", "w") as f:
        f.write("\n".join(t.replace("\n", " ") for t in texts) + "\n")
    subprocess.run([f"{core}/target/release/examples/clause_factors", f"{work}/{name}.txt", f"{work}/{name}.jsonl"], cwd=core, check=True)
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
    pairs += [(kind, b, a) for b, a in es[OFFSET:OFFSET + PER]]
sides = parse([t for _, b, a in pairs for t in (b, a)], "cands")
all_rule = parse(rule_text[:5000], "rule")
word = lambda t: re.findall(r"\w+(?:'\w+)?", t.lower())
VERBS = {"is", "are", "was", "were", "be", "been", "has", "have", "had"}


def group(b, a):
    x, y = word(b), word(a)
    d = [(p, q) for p, q in zip(x, y) if p != q] if len(x) == len(y) else []
    return "be/have swap" if len(d) == 1 and d[0][0] in VERBS and d[0][1] in VERBS else "other"


def pipeline(source_ids, n_tokens, K, encoding):
    g = GrammarSpec(roles=[f"t{i}" for i in range(n_tokens)], allowed_transitions={i: list(range(n_tokens)) for i in range(n_tokens)},
                    forbidden_tuples=set(), benign_tuples=set(), tuple_size=K, n_qubits=K * 4 + 2, label="clause", encoding=encoding)
    return HandoffPipeline(g, max_benign_refs=30, classical_nu=0.1, seed=42).fit(source_ids)


for n_rule, combo in itertools.product((1000, 5000), COMBOS):
    K = len(combo)
    rule = all_rule[:n_rule]
    tup = lambda cl: [tuple(c[f] for f in combo) for c in cl]
    ids = {}
    tid = lambda f, v: ids.setdefault((f, v), len(ids))
    enc = lambda cl: [tid(f, c[f]) for c in cl for f in combo]
    counts = collections.Counter(t for cl in rule for t in tup(cl))
    idx = []
    for i in range(len(pairs)):
        b, a = tup(sides[2 * i]), tup(sides[2 * i + 1])
        if b and a and b != a and any(counts[t] == 0 for t in b) and any(counts[t] == 0 for t in a):
            idx.append(i)
    source = [s for s in map(enc, rule) if s]
    cands = [enc(sides[j]) for i in idx for j in (2 * i, 2 * i + 1)]
    if len(idx) < 30:
        print(f"{combo} rulebook {n_rule}: only {len(idx)} edits; skipped", flush=True)
        continue
    # Description tables: a value's vector from its attributes, unknown values get
    # the reserved last dimension.
    inverse = {i: k for k, i in ids.items()}
    tables = [{} for _ in range(K)]
    for r, f in enumerate(combo):
        for i, (ff, v) in inverse.items():
            if ff == f and (vec := vector(f, v)):
                tables[r][i] = vec
    encoding = {"type": "role_embedding_v1", "k": K, "dims": [DIM] * K, "tables": tables, "unknown": [[0.0] * (DIM - 1) + [1.0]] * K, "dim": DIM, "one_hot_roles": []}
    described = pipeline(source, len(ids), K, encoding)
    learned = pipeline(source, len(ids), K, {"type": "role_embedding_v1", "dim": DIM, "one_hot_roles": []})
    fd, fl = described.tuple_scores(cands), learned.tuple_scores(cands)
    # Classical attribute pairs.
    pair, single, tot = collections.Counter(), collections.Counter(), 0
    bag = lambda t: [(p, a) for p, (f, v) in enumerate(zip(combo, t)) for a in attrs(f, v)]
    for cl in rule:
        for t in tup(cl):
            b = bag(t)
            single.update(b); tot += 1
            pair.update((x, y) for x, y in itertools.combinations(b, 2) if x[0] != y[0])

    def pmi(t):
        vals = [math.log(((pair[(x, y)] + 0.5) * max(tot, 1)) / ((single[x] + 1) * (single[y] + 1)))
                for x, y in itertools.combinations(bag(t), 2) if x[0] != y[0]]
        return -sum(vals) / len(vals) if vals else 0.0

    stat = collections.defaultdict(lambda: collections.defaultdict(list))
    for m, i in enumerate(idx):
        b, a = tup(sides[2 * i]), tup(sides[2 * i + 1])
        g = stat[group(pairs[i][1], pairs[i][2])]
        changed = [j for j, t in enumerate(b) if t not in a]
        g["chance"].append(len(changed) / len(b))
        for name, sb, sa in (("description", fd[2 * m], fd[2 * m + 1]), ("learned", fl[2 * m], fl[2 * m + 1]),
                             ("attributes", [pmi(t) for t in b], [pmi(t) for t in a]),
                             ("counts", [-counts[t] for t in b], [-counts[t] for t in a])):
            g[name].append((max(sa) < max(sb)) + 0.5 * (max(sa) == max(sb)))
            worst = [j for j, v in enumerate(sb) if v == max(sb)]
            g[name + " loc"].append(len(set(worst) & set(changed)) / len(worst))
    avg = lambda x: sum(x) / max(len(x), 1)
    for gname in sorted(stat):
        g = stat[gname]
        print(f"{combo} rulebook {n_rule} {gname} (n={len(g['counts'])}): " + ", ".join(f"{n} {avg(g[n]):.3f}/{avg(g[n + ' loc']):.3f}" for n in ("description", "learned", "attributes", "counts"))
              + f"; chance loc {avg(g['chance']):.3f}; rank desc {described._T.shape[1]} learned {learned._T.shape[1]}", flush=True)
    with open(f"{work}/judge_{combo}_{n_rule}.tsv", "w") as f:
        for i in idx:
            k, b, a = pairs[i]
            f.write(f"{group(b, a)}\t{i}\t{b}\t{a}\n")
