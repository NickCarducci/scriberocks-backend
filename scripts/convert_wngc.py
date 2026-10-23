"""The Princeton WordNet Gloss Corpus (WordNet's definitions and examples,
sense-tagged; WordNet license) to the WSD Evaluation Framework's format
(data.xml + gold.key.txt), the format src/semcor.rs reads, so SemCor's
tested reader and sense-key mapping handle it unchanged. Only hand-tagged
words (tag="man") become instances; every sense key they carry counts as an
acceptable answer, as in SemCor.

usage: python3 -I convert_wngc.py <WordNet-3.0/glosstag/merged> <out prefix>
"""
import os, sys, xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

merged, out = sys.argv[1:3]
POS = {"1": "NOUN", "2": "VERB", "3": "ADJ", "4": "ADV", "5": "ADJ"}
sentences = instances = tagged_words = 0
with open(out + ".data.xml", "w") as x, open(out + ".gold.key.txt", "w") as g:
    x.write('<?xml version="1.0" encoding="UTF-8" ?>\n<corpus lang="en" source="wngc">\n<text id="wngc" source="wordnet-gloss-corpus">\n')
    for name in ("noun", "verb", "adj", "adv"):
        for _, syn in ET.iterparse(os.path.join(merged, f"{name}.xml"), events=("end",)):
            if syn.tag != "synset": continue
            for gloss in syn.findall("gloss"):
                if gloss.get("desc") != "wsd": continue
                for k, part in enumerate(list(gloss)):
                    if part.tag not in ("def", "ex"): continue
                    words = [w for w in part.iter("wf")]
                    if not words: continue
                    sid = f"{syn.get('id')}.{part.tag}{k}"
                    x.write(f'<sentence id="{sid}">\n'); sentences += 1
                    for t, w in enumerate(words):
                        text = "".join(w.itertext()).strip()
                        if not text: continue
                        keys = [i.get("sk") for i in w.findall("id") if i.get("sk")]
                        if w.get("tag") == "man" and keys:
                            pos = POS.get(keys[0].split("%")[1][:1], "X")
                            lemma = w.find("id").get("lemma", text.lower())
                            iid = f"{sid}.t{t}"
                            x.write(f'<instance id="{iid}" lemma={quoteattr(lemma)} pos="{pos}">{escape(text)}</instance>\n')
                            g.write(f"{iid} {' '.join(keys)}\n"); instances += 1
                        else:
                            lemma = (w.get("lemma") or text).split("|")[0].split("%")[0]
                            x.write(f'<wf lemma={quoteattr(lemma)} pos="X">{escape(text)}</wf>\n')
                    x.write("</sentence>\n")
            syn.clear()
    x.write("</text>\n</corpus>\n")
print(f"{sentences} definition/example sentences, {instances} hand-tagged words")
