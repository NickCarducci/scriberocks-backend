"""Edit records for FidelityAI: what a normal editor's edit looks like.

usage: python3 -I fidelity_edit_records.py <simplewiki-history dir> <out dir> [source n = 20000] [test n = 1000]

From Simple English Wikipedia sentence edits (extract_simplewiki_edits.py),
keeps edits that make one contiguous change and writes each as a record:
  {"op": insert|delete|replace, "removed": ..., "inserted": ..., "left": ..., "right": ...}
(lowercased words; "" when empty; left/right = the neighboring word or <s>/</s>).
  source.jsonl      normal edits from training articles
  test_real.jsonl   real edits from held-out articles
  test_random.jsonl the same edits displaced: the same change applied at another
                    spot in the same sentence (only placement differs)
"""
import collections, difflib, json, random, re, sys

src_dir, out = sys.argv[1], sys.argv[2]
n_source = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
TOKEN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")
rng = random.Random(11)


def tokens(s):
    return [t.lower() for t in TOKEN.findall(s)]


def record(a, b):
    """The record for an edit with exactly one contiguous change, else None."""
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    if len(ops) != 1:
        return None
    op, i1, i2, j1, j2 = ops[0]
    return {
        "op": op,
        "removed": " ".join(a[i1:i2]),
        "inserted": " ".join(b[j1:j2]),
        "left": a[i1 - 1] if i1 > 0 else "<s>",
        "right": a[i2] if i2 < len(a) else "</s>",
    }


def pairs(path, n):
    out = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            a, b = tokens(parts[1]), tokens(parts[2])
            r = record(a, b)
            if r:
                out.append((a, r))
            if len(out) >= n:
                break
    return out


def apply_record(a, r):
    """The edited sentence a record describes (its first matching spot)."""
    k = len(r["removed"].split()) if r["removed"] else 0
    left = [r["left"]] if r["left"] != "<s>" else []
    for i in range(len(a) + 1 - k):
        if a[i:i + k] == (r["removed"].split() if k else []) and (not left or a[i - 1:i] == left):
            return a[:i] + (r["inserted"].split() if r["inserted"] else []) + a[i + k:]
    return a


source = pairs(f"{src_dir}/train.tsv", n_source)
held = pairs(f"{src_dir}/test.tsv", n_test)
inserted = collections.Counter(r["inserted"] for _, r in source if r["inserted"])
common = [p for p, _ in inserted.most_common(200)]

# Negatives: each real held-out edit, displaced. The same content applied at
# another spot in the same sentence (insert the same words elsewhere; delete
# or replace a different span of the same length), so word frequency cancels
# and only "does this edit fit here" is tested.
randoms = []
sentence_pairs = [("1", a, apply_record(a, r)) for a, r in held]
for a, r in held:
    k = len(r["removed"].split()) if r["removed"] else 0
    spots = [i for i in range(len(a) + 1 - k) if a[i:i + k] != r["removed"].split() or k == 0]
    spots = [i for i in spots if not (r["op"] == "insert" and a[i - 1:i] == [r["left"]] and a[i:i + 1] == [r["right"]])]
    if not spots:
        continue
    i = rng.choice(spots)
    b = a[:i] + (r["inserted"].split() if r["inserted"] else []) + a[i + k:]
    d = record(a, b)
    if d and d != r:
        randoms.append(d)
        sentence_pairs.append(("0", a, b))

# Whole sentences too (label 1 = real, 0 = displaced; before, after), for
# judges that read the sentence rather than the record.
with open(f"{out}/test_sentences.tsv", "w") as f:
    for label, a, b in sentence_pairs:
        f.write(f"{label}\t{' '.join(a)}\t{' '.join(b)}\n")

for name, rows in [("source", [r for _, r in source]), ("test_real", [r for _, r in held]), ("test_random", randoms)]:
    with open(f"{out}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} records")
