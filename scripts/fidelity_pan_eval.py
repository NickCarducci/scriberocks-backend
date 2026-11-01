"""Does a piece agree with itself, on real multi-author text? PAN 2025 Multi-Author
Writing Style Analysis (Reddit comments combined into documents, CC BY 4.0, Zenodo
14891299; local evaluation only): every boundary between consecutive sentences is
labelled author change or not, at three levels of topic control (easy: topics vary,
medium: little variety, hard: one topic).

Scored on the validation split, rulebook and thresholds from the train split:
  boundary: AUC of "this boundary is an author change" (pooled, and within document)
            and macro F1, PAN's measure, at a threshold set on train;
  window:   five consecutive sentences with no change against five with at least one
            (the real version of the spliced control), AUC.
Self-fidelity (|<a|b>|^2 of sentence states in embeddings learned from the rulebook,
SVD's first component removed) against agreeing values, adjacent tuple-pair counts,
length difference, and two word-level measures that see far more than closed choices
(function-word and character-trigram cosine).

usage: ~/.venvs/fidelityai/bin/python -I fidelity_pan_eval.py <core> <engine> <pan dir> <work> [train docs per level = 1000]
"""
import collections, glob, json, math, os, random, re, subprocess, sys

import numpy as np

core, engine, pan, work = sys.argv[1:5]
n_train = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
sys.path.insert(0, engine)
import sklearn.decomposition as sd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from fidelityai.role_embedding import learn_role_embeddings, record_features  # noqa: E402

FUNCTION = set("the a an and or but if so as of to in on at by for with from that this these those it its i you he she we they me him her us them my your his our their is are was were be been am do does did have has had will would can could should may might must not no n't very just really also then than there here who what which when where why how".split())


class DropFirst(sd.TruncatedSVD):
    """SVD's first component is what all a role's values share: removing it leaves their differences."""

    def __init__(self, n_components=2, random_state=None):
        super().__init__(n_components=n_components + 1, random_state=random_state)

    def fit_transform(self, X, y=None):
        if min(X.shape) <= self.n_components:
            self.n_components = min(X.shape) - 1
        u = super().fit_transform(X)
        return u[:, 1:] if u.shape[1] > 1 else u


def load(level, split, limit=None):
    docs = []
    for path in sorted(glob.glob(f"{pan}/{level}/{split}/problem-*.txt"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
        with open(path, newline="") as f:
            lines = f.read().split("\n")
        lines = [l for l in lines if l.strip()]
        changes = json.load(open(path.replace("problem-", "truth-problem-").replace(".txt", ".json")))["changes"]
        if len(changes) == len(lines) - 1:
            docs.append((lines, changes))
    return docs[:limit]


def parse(docs, name):
    """Per document, per line: the line's sentences' closed choices (a line can parse as several)."""
    out = f"{work}/{name}.jsonl"
    if not os.path.exists(out):
        with open(f"{work}/{name}.txt", "w") as f:
            for lines, _ in docs:
                for l in lines:
                    f.write(l.replace("\r", " ") + "\n")
        subprocess.run([f"{core}/target/release/examples/consistency_features", f"{work}/{name}.txt", out], cwd=core, check=True)
    rows = [json.loads(l) for l in open(out)]
    parsed, at = [], 0
    for lines, _ in docs:
        parsed.append(rows[at:at + len(lines)]); at += len(lines)
    assert at == len(rows), (at, len(rows))
    return parsed


def auc(y, s):
    return roc_auc_score(y, s) if 0 < sum(y) < len(y) else float("nan")


def macro_f1(y, s, t):
    y, p = np.asarray(y), np.asarray(s) >= t
    def f1(a, b):
        tp = (a & b).sum()
        return 2 * tp / max(a.sum() + b.sum(), 1)
    return (f1(y == 1, p) + f1(y == 0, ~p)) / 2


def best_threshold(y, s):
    grid = np.quantile(s, np.linspace(0.02, 0.98, 49))
    return max(np.unique(grid), key=lambda t: macro_f1(y, s, t))


words = lambda l: re.findall(r"[a-z']+", l.lower())


def cosine(a, b):
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return sum(v * b[k] for k, v in a.items() if k in b) / (na * nb) if na and nb else 0.0


func = lambda l: collections.Counter(w for w in words(l) if w in FUNCTION)
tri = lambda l: collections.Counter(l.lower()[i:i + 3] for i in range(len(l) - 2))


class Variant:
    """A factor set's sentence states, with the shared component kept or removed and two-valued roles embedded or one-hot."""

    def __init__(self, factors, rule, drop, hot_binary):
        self.factors, self.ids = factors, {}
        tuples = [self.enc(c) for d in rule for line in d for c in line]
        roles = [{t[i] for t in tuples} for i in range(len(factors))]
        hot = [i for i, v in enumerate(roles) if hot_binary and len(v) <= 2]
        svd = sd.TruncatedSVD
        sd.TruncatedSVD = DropFirst if drop else svd
        try:
            self.g = learn_role_embeddings(tuples, len(factors), dim={1: 16, 2: 16, 3: 16, 4: 8}[len(factors)], one_hot_roles=hot)
        finally:
            sd.TruncatedSVD = svd
        self.states = {}

    def enc(self, c):
        return tuple(self.ids.setdefault(f"{f}:{c[f]}", len(self.ids)) for f in self.factors)

    def state(self, c):
        t = self.enc(c)
        if t not in self.states:
            self.states[t] = record_features([t], self.g)
        return self.states[t]

    def overlap(self, a, b):
        return float(self.state(a) @ self.state(b)) ** 2


def report_roles(v, label):
    for i, f in enumerate(v.factors):
        table = [np.array(x) for x in v.g["tables"][i].values()]
        o = [float(a @ b) ** 2 for n, a in enumerate(table) for b in table[n + 1:]]
        print(f"    {label} role {f}: {len(table)} values, dimension {v.g['dims'][i]}, overlap between values {min(o) if o else 1:.3f}-{max(o) if o else 1:.3f}", flush=True)


REPEATED = set()  # lines in three or more of a level's documents: moderator boilerplate, not anyone's writing


def repeated_lines(level):
    seen = collections.Counter()
    for split in ("train", "validation"):
        for lines, _ in load(level, split):
            seen.update(set(lines))
    return {l for l, n in seen.items() if n >= 3}


def boundaries(docs, parsed):
    """(doc, line a, line b, last sentence of a, first of b, label) where both lines parsed and neither is boilerplate."""
    for n, ((lines, ch), p) in enumerate(zip(docs, parsed)):
        for i, y in enumerate(ch):
            if p[i] and p[i + 1] and lines[i] not in REPEATED and lines[i + 1] not in REPEATED:
                yield n, lines[i], lines[i + 1], p[i][-1], p[i + 1][0], y


def windows(docs, parsed, size=5):
    """Consecutive non-overlapping windows of real sentences: (sentences, lines, has a change)."""
    for (lines, ch), p in zip(docs, parsed):
        for i in range(0, len(lines) - size + 1, size):
            cs = [c for line in p[i:i + size] for c in line[:1]]
            if len(cs) == size and not REPEATED.intersection(lines[i:i + size]):
                yield cs, lines[i:i + size], int(any(ch[i:i + size - 1]))


def run(level, pool_rule):
    train, val = load(level, "train", n_train), load(level, "validation")
    tp, vp = parse(train, f"{level}_train_{n_train}"), parse(val, f"{level}_val")
    tb, vb = list(boundaries(train, tp)), list(boundaries(val, vp))
    print(f"\n{level}: validation {len(val)} documents, {len(vb)} boundaries, {sum(b[5] for b in vb) / len(vb):.3f} changes; rulebook {len(pool_rule)} documents", flush=True)
    vy, ty = [b[5] for b in vb], [b[5] for b in tb]
    single_runs = pool_rule
    keys = lambda d: [(a, b) for line_a, line_b in zip(d, d[1:]) for a in line_a[-1:] for b in line_b[:1]]
    results, chosen = {}, {}

    def record(name, fn, window_fn=None):
        """fn(boundary) is higher when the boundary looks like a change."""
        sv, st = np.array([fn(b) for b in vb]), np.array([fn(b) for b in tb])
        t = best_threshold(ty, st)
        within = [auc(*zip(*[(b[5], s) for b, s in zip(vb, sv) if b[0] == n])) for n in sorted({b[0] for b in vb})]
        w = ""
        if window_fn:
            ws = list(windows(val, vp))
            w = f", window {auc([x[2] for x in ws], [window_fn(x) for x in ws]):.3f}"
        results[name] = sv
        chosen[name] = auc(ty, st)
        print(f"  {name}: train {chosen[name]:.3f}, boundary {auc(vy, sv):.3f}, within document {np.nanmean(within):.3f}, macro F1 {macro_f1(vy, sv, t):.3f}{w}", flush=True)

    def pairs_mean(f, cs):
        return float(np.mean([f(a, b) for n, a in enumerate(cs) for b in cs[n + 1:]]))

    for factors in ("TPCV", "TPV", "TP"):
        tup = lambda c: tuple(c[f] for f in factors)
        adj = collections.Counter((tup(a), tup(b)) for d in single_runs for a, b in keys(d))
        record(f"{factors} agreeing values", lambda b: -sum(b[3][f] == b[4][f] for f in factors),
               lambda w: sum(len({c[f] for c in w[0]}) for f in factors))
        record(f"{factors} adjacent tuple pairs", lambda b: -math.log1p(adj[(tup(b[3]), tup(b[4]))]),
               lambda w: -float(np.mean([math.log1p(adj[(tup(a), tup(b))]) for a, b in zip(w[0], w[0][1:])])))
        for label, drop, hot in (("no removal", False, False), ("removal", True, False), ("removal, two-valued one-hot", True, True)):
            v = Variant(factors, single_runs, drop, hot)
            if level == "easy":
                report_roles(v, label)
            record(f"{factors} self-fidelity, {label}", lambda b: -v.overlap(b[3], b[4]), lambda w: -pairs_mean(v.overlap, w[0]))
    record("length difference", lambda b: abs(len(words(b[1])) - len(words(b[2]))),
           lambda w: float(np.std([len(words(l)) for l in w[1]])))
    record("function-word cosine", lambda b: -cosine(func(b[1]), func(b[2])),
           lambda w: -pairs_mean(lambda a, b: cosine(func(a), func(b)), w[1]))
    record("character-trigram cosine", lambda b: -cosine(tri(b[1]), tri(b[2])),
           lambda w: -pairs_mean(lambda a, b: cosine(tri(a), tri(b)), w[1]))
    return results, chosen, vb


if __name__ == "__main__":
    # Rulebook: the same level's train documents, so the rulebook is this register.
    for level in ("easy", "medium", "hard"):
        REPEATED.clear(); REPEATED.update(repeated_lines(level))
        rule = parse(load(level, "train", n_train), f"{level}_train_{n_train}")
        results, chosen, vb = run(level, rule)
        # Bootstrap over documents: the closed-choice self-fidelity chosen on train against each comparison.
        docs = sorted({b[0] for b in vb})
        y = np.array([b[5] for b in vb]); at = collections.defaultdict(list)
        for i, b in enumerate(vb):
            at[b[0]].append(i)
        r = random.Random(7)
        best = max((k for k in results if "self-fidelity" in k), key=chosen.get)
        for other in ("TPCV agreeing values", "function-word cosine", "character-trigram cosine"):
            d = []
            for _ in range(300):
                idx = [i for n in r.choices(docs, k=len(docs)) for i in at[n]]
                d.append(auc(y[idx], results[best][idx]) - auc(y[idx], results[other][idx]))
            print(f"  {best} minus {other}: {np.mean(d):+.3f} (95% {np.quantile(d, .025):+.3f} to {np.quantile(d, .975):+.3f})", flush=True)
