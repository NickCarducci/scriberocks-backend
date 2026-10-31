"""FidelityAI on structure alone: no vocabulary, only each sentence's dependency
relations and punctuation marks (examples/structure_parse.rs), as sliding
windows of k = 3, role embeddings. Real data only, local and free.

  edits   Simple English Wikipedia corrections that change structure: comma
          edits, word reordering, sentence splits and merges, clause rewrites.
          Rulebook: 5,000 stable sentences. Does the after score more typical?
          Splits and merges are rerun on a rulebook with sentence boundaries.
  rocks   certified against ordinary rock-sized passages (100-150 characters,
          length-matched); rulebook 5,000 certified passages. AUC.
  authors one Tatoeba contributor's sentences against others'; rulebook 2,000
          of theirs. AUC.
Reported: FidelityAI's record score and worst tuple, and a plain count baseline
(mean log-probability of the relation trigrams under the rulebook). Writes
before/after pairs for the judge (text_judge_eval).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_structure_eval.py <core dir> <engine dir> <work dir> [per kind = 400]
"""
import bz2, collections, json, math, random, re, subprocess, sys, types, zlib

core, engine, work = sys.argv[1], sys.argv[2], sys.argv[3]
per_kind = int(sys.argv[4]) if len(sys.argv) > 4 else 400
K = 3
sys.path.insert(0, engine)
from fidelityai.quantum import abcd_fit_and_score  # noqa: E402

rng = random.Random(21)


def parse(texts, name):
    with open(f"{work}/{name}.txt", "w") as f:
        f.write("\n".join(t.replace("\n", " ") for t in texts) + "\n")
    subprocess.run([f"{core}/target/release/examples/structure_parse", f"{work}/{name}.txt", f"{work}/{name}.jsonl"], cwd=core, check=True)
    return [json.loads(l) for l in open(f"{work}/{name}.jsonl")]


ids = {}
stream = lambda sents: [ids.setdefault(t, len(ids)) for s in sents for t in ["<s>"] + s]
grams = lambda sents: (lambda x: [x[i:i + K] for i in range(len(x) - K + 1)])(stream(sents))
flat = lambda sents: [t for g in grams(sents) for t in g]


def score(rulebook, cands):
    """Per candidate: typicality by record, by worst tuple, by plain trigram counts."""
    for d in rulebook + cands:
        stream(d)
    counts = collections.Counter(tuple(g) for d in rulebook for g in grams(d))
    total, n = sum(counts.values()), len(counts) + 1
    count = [sum(math.log((counts[tuple(g)] + 1) / (total + n)) for g in grams(d)) / max(len(grams(d)), 1) for d in cands]
    res = abcd_fit_and_score(source_dataset=types.SimpleNamespace(episodes=[[{"sequence": flat(d)}] for d in rulebook if grams(d)], label="structure"),
                             artifact_dataset=types.SimpleNamespace(episodes=[[{"sequence": flat(d)}] for d in cands], label="candidates"),
                             max_qubits=20, seed=42, encoding={"type": "role_embedding_v1", "dim": 16, "one_hot_roles": []}, k_values=[K])
    pipe = res["pipe"]
    assert len(res["mapped_artifacts"]) == len(cands)
    worst = [-max(t) if t else None for t in pipe.tuple_scores(res["mapped_artifacts"])]
    return {"record": [None if s is None else -s for s in res["artifact_scores"]], "worst tuple": worst, "counts": count}


def auc(xs, ys):
    return sum((x > y) + 0.5 * (x == y) for x in xs for y in ys) / max(len(xs) * len(ys), 1)


# ---- Edits
word = lambda s: re.findall(r"\w+(?:'\w+)?", s.lower())
marks = lambda s: re.findall(r"[,;:]", s)
ENDS = re.compile(r"(?<=[a-z]{4})[.!?](?=\s)")
JOIN = {"and", "but", "so", "which", "because", "while"}


def kind(b, a):
    wb, wa = word(b), word(a)
    if min(len(wb), len(wa)) < 6:
        return None
    if wb == wa:
        nb, na = len(ENDS.findall(b)), len(ENDS.findall(a))
        if abs(nb - na) == 1:
            return "split" if na > nb else "merge"
        return "comma" if marks(b) != marks(a) else None
    if sorted(wb) == sorted(wa):
        return "reorder"
    cb, ca = collections.Counter(wb), collections.Counter(wa)
    gone, new = sum((cb - ca).values()), sum((ca - cb).values())
    if gone <= 1 and new == 0 and set((cb - ca)) <= JOIN and len(ENDS.findall(a)) == len(ENDS.findall(b)) + 1:
        return "split"
    if new <= 1 and gone == 0 and set((ca - cb)) <= JOIN and len(ENDS.findall(b)) == len(ENDS.findall(a)) + 1:
        return "merge"
    if 3 <= gone + new <= 0.4 * min(len(wb), len(wa)) and gone and new:
        return "rewrite"
    return None


edits = collections.defaultdict(list)
for f in ("train", "dev", "test"):
    for line in open(f"{core}/data/simplewiki-history/{f}.tsv"):
        c = line.rstrip("\n").split("\t")
        if len(c) == 3 and (k := kind(c[1], c[2])):
            edits[k].append((c[1], c[2]))
print("edits found: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(edits.items())), flush=True)
pairs = []
for k, v in sorted(edits.items()):
    rng.shuffle(v)
    pairs += [(k, b, a) for b, a in v[:per_kind]]
stable = [l.rstrip("\n").split("\t")[3] for l in open(f"{core}/data/simplewiki-history/stable_train.tsv")]
rule = parse(rng.sample(stable, 5000), "stable")
sides = parse([t for _, b, a in pairs for t in (b, a)], "edits")
keep = [i for i in range(len(pairs)) if grams(sides[2 * i]) and grams(sides[2 * i + 1])]
cands = [sides[j] for i in keep for j in (2 * i, 2 * i + 1)]
kept = [pairs[i] for i in keep]
with open(f"{work}/judge_pairs.tsv", "w") as f:
    for n, (k, b, a) in enumerate(kept):
        f.write(f"{k}\t{n}\t0\t{b}\n{k}\t{n}\t1\t{a}\n")
s = score(rule, cands)
for name, vals in s.items():
    by = collections.defaultdict(list)
    for i, (k, _, _) in enumerate(kept):
        b, a = vals[2 * i], vals[2 * i + 1]
        if a is not None and b is not None:
            by[k].append((a > b) + 0.5 * (a == b))
    print(f"edits, {name}: " + ", ".join(f"{k} {sum(v) / len(v):.3f} ({len(v)})" for k, v in sorted(by.items())), flush=True)

# Splits and merges again, against a rulebook that has sentence boundaries:
# stable sentences alone would make any boundary look unseen.
def consecutive():
    out = []
    for t, _, p in [l.rstrip("\n").split("\t") for l in open(f"{core}/data/simplewiki-quality/paragraphs.tsv")]:
        x = [y.strip() for y in re.split(r"(?<=[.!?])\s+", p) if y.strip()]
        out += [" ".join(x[i:i + 2]) for i in range(0, len(x) - 1, 2)] + x
    return out


mixed = parse(random.Random(4).sample(consecutive(), 5000), "mixed_rule")
sm = [i for i, (k, _, _) in enumerate(kept) if k in ("split", "merge")]
s = score(mixed, [cands[j] for i in sm for j in (2 * i, 2 * i + 1)])
for name, vals in s.items():
    by = collections.defaultdict(list)
    for m, i in enumerate(sm):
        b, a = vals[2 * m], vals[2 * m + 1]
        if a is not None and b is not None:
            by[kept[i][0]].append((a > b) + 0.5 * (a == b))
    print(f"split/merge with boundaries in the rulebook, {name}: " + ", ".join(f"{k} {sum(v) / len(v):.3f} ({len(v)})" for k, v in sorted(by.items())), flush=True)

# Control: clean text, one sentence against two. FidelityAI's scores react to the
# boundary itself (worst tuple 1.000, record 0.301; counts 0.500), so its split
# and merge results above are not evidence of quality.
paragraphs = [l.rstrip("\n").split("\t")[2] for l in open(f"{core}/data/simplewiki-quality/paragraphs.tsv")]
random.Random(8).shuffle(paragraphs)
ones, twos = [], []
for p in paragraphs[:900]:
    x = [y.strip() for y in re.split(r"(?<=[.!?])\s+", p) if y.strip()]
    if len(x) >= 3:
        ones.append(x[0]); twos.append(" ".join(x[1:3]))
c1, c2 = parse(ones[:300], "ctl1"), parse(twos[:300], "ctl2")
s = score(mixed, c1 + c2)
for name, vals in s.items():
    print(f"control, one clean sentence more typical than two, {name}: {auc([v for v in vals[:len(c1)] if v is not None], [v for v in vals[len(c1):] if v is not None]):.3f}", flush=True)

# ---- Rocks
split = lambda p: [x.strip() for x in re.split(r"(?<=[.!?])\s+", p) if x.strip()]
held = lambda title: zlib.crc32(title.encode()) % 10 < 3


def passages(paragraph, lo=100, hi=150):
    out, run = [], []
    for x in split(paragraph):
        run.append(x)
        text = " ".join(run)
        if len(text) > hi:
            run = [x] if len(x) <= hi else []
            text = " ".join(run)
        if lo <= len(text) <= hi:
            out.append(text); run = []
    return out


q = f"{core}/data/simplewiki-quality"
cert = [l.rstrip("\n").split("\t") for l in open(f"{q}/paragraphs.tsv")]
ord_ = [l.rstrip("\n").split("\t") for l in open(f"{q}/ordinary.tsv")]
r2 = random.Random(5)
train = [x for t, _, p in cert if not held(t) for x in passages(p)]
good = [x for t, _, p in cert if held(t) for x in passages(p)]
plain = list(dict.fromkeys(x for _, p in ord_ for x in passages(p)))
r2.shuffle(train); r2.shuffle(good); r2.shuffle(plain)
n = min(len(good), len(plain), 600)
rb, cn = parse(train[:5000], "rock_rule"), parse(good[:n] + plain[:n], "rock_cand")
ok = [i for i in range(len(cn)) if grams(cn[i])]
s = score(rb, [cn[i] for i in ok])
for name, vals in s.items():
    p = [v for i, v in zip(ok, vals) if i < n and v is not None]
    o = [v for i, v in zip(ok, vals) if i >= n and v is not None]
    print(f"rocks, {name}: AUC {auc(p, o):.3f} ({len(p)} certified, {len(o)} ordinary)", flush=True)

# ---- Authors
by_author = collections.defaultdict(list)
for line in bz2.open(f"{core}/data/tatoeba/eng_sentences_detailed.tsv.bz2", "rt"):
    c = line.rstrip("\n").split("\t")
    if len(c) >= 4 and 5 <= len(c[2].split()) <= 25:
        by_author[c[3]].append(c[2])
for author in ["CK", "Amastan", "OsoHombre", "CH", "Hybrid", "CM"]:
    r3 = random.Random(7)
    own = list(dict.fromkeys(by_author[author])); r3.shuffle(own)
    others = [x for a, ss in by_author.items() if a != author for x in ss]
    rb = parse(own[:2000], "author_rule")
    cn = parse(own[2000:2300] + r3.sample(others, 300), "author_cand")
    s = score(rb, cn)
    print(f"author {author}: " + ", ".join(f"{name} {auc([v for v in vals[:300] if v is not None], [v for v in vals[300:] if v is not None]):.3f}" for name, vals in s.items()), flush=True)
