"""ProLex substitutions as rows for examples/substitution_records.rs (same
columns as swords_substitutions.py): split, target id, label, prefix, target,
substitute, suffix. Label 1 = acceptable, 0 = unacceptable (human-judged in
learners' essays). Evaluation only: the TOEFL-11 sentences are licensed by
ETS, so never in training, rulebooks or anything shipped.

usage: python3 -I prolex_substitutions.py <prolex dir> <out tsv>
"""
import ast, csv, hashlib, re, sys

prolex, out = sys.argv[1], sys.argv[2]
clean = lambda s: " ".join(s.split())
with open(out, "w") as f:
    for split in ("dev", "test"):
        for row in csv.DictReader(open(f"{prolex}/ProLex_v1.0_{split}.csv")):
            m = re.search(r"\*\*(.+?)\*\*", row["Sentence"])
            if not m:
                continue
            prefix, target, suffix = row["Sentence"][:m.start()], m.group(1), row["Sentence"][m.end():]
            tid = "p:" + hashlib.sha1(row["Sentence"].encode()).hexdigest()[:16]
            for y, key in ((1, "acc_subs"), (0, "unacc_subs")):
                for w in ast.literal_eval(row[key] or "[]"):
                    if " " not in w:
                        f.write("\t".join(["prolex-" + split, tid, str(y), clean(prefix), target, w, clean(suffix)]) + "\n")
