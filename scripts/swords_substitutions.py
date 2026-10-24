"""SWORDS substitutions as rows for examples/substitution_records.rs.

Each judged one-word substitute, with its sentence split around the target:
split, target id, label, prefix, target, substitute, suffix. Label 1 =
accepted (more than half the raters), 0 = rejected (no rater accepted);
substitutes in between are left out.

usage: python3 -I swords_substitutions.py <swords dir> <out tsv>
"""
import json, sys

swords, out = sys.argv[1], sys.argv[2]
with open(out, "w") as f:
    for split in ("dev", "test"):
        d = json.load(open(f"{swords}/swords-v1.1_{split}.json"))
        for sid, sub in d["substitutes"].items():
            labels = [l for l in d["substitute_labels"].get(sid, []) if l != "UNSURE"]
            w = sub["substitute"]
            if not labels or " " in w:
                continue
            yes = sum(l.startswith("TRUE") for l in labels) / len(labels)
            y = 1 if yes > 0.5 else 0 if yes == 0 else None
            if y is None:
                continue
            t = d["targets"][sub["target_id"]]
            ctx = d["contexts"][t["context_id"]]["context"]
            off, tw = t["offset"], t["target"]
            ends = lambda i: ctx[i] == "\n" or (ctx[i] in ".!?" and (i + 1 >= len(ctx) or ctx[i + 1].isspace()))
            start = next((i + 1 for i in range(off - 1, -1, -1) if ends(i)), 0)
            end = next((i + 1 for i in range(off + len(tw), len(ctx)) if ends(i)), len(ctx))
            clean = lambda s: " ".join(s.split())
            f.write("\t".join([split, sub["target_id"], str(y), clean(ctx[start:off]), tw, w, clean(ctx[off + len(tw):end])]) + "\n")
