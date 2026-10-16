//! SemCor in the unified WSD framework format (Raganato et al. 2017):
//! natural text whose content words carry WordNet 3.0 sense keys. Used to
//! evaluate sense choice at natural frequencies and to add sense examples.
//!
//! SemCor is © Princeton University under the WordNet license (permissive,
//! notice must accompany copies). The framework's other evaluation sets are
//! only read locally, never redistributed.

use crate::lexicon::{Lexicon, Pos};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    /// Position of the tagged word in `TaggedSentence::words`.
    pub index: usize,
    pub lemma: String,
    pub pos: Pos,
    /// Gold sense keys; any of them counts as correct.
    pub keys: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TaggedSentence {
    /// Source document ("d000"), for splitting by document.
    pub document: String,
    pub words: Vec<String>,
    pub instances: Vec<Instance>,
}

/// Reads `[dataset].data.xml` and `[dataset].gold.key.txt`.
pub fn parse(xml: &str, gold: &str) -> Vec<TaggedSentence> {
    let gold: HashMap<&str, Vec<String>> = gold
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?, it.map(String::from).collect()))
        })
        .collect();
    let mut out = Vec::new();
    let mut current: Option<TaggedSentence> = None;
    for line in xml.lines().map(str::trim) {
        if line.starts_with("<sentence ") {
            let id = attr(line, "id").unwrap_or_default();
            let document = id.split('.').next().unwrap_or_default().to_string();
            current = Some(TaggedSentence { document, words: vec![], instances: vec![] });
        } else if line.starts_with("</sentence>") {
            out.extend(current.take());
        } else if let Some(s) = current.as_mut() {
            let Some(text) = element_text(line) else { continue };
            if line.starts_with("<instance ") {
                let pos = attr(line, "pos").as_deref().and_then(Pos::from_upos);
                let id = attr(line, "id").unwrap_or_default();
                if let (Some(pos), Some(keys)) = (pos, gold.get(id.as_str())) {
                    let lemma = attr(line, "lemma").unwrap_or_default().to_lowercase();
                    s.instances.push(Instance { index: s.words.len(), lemma, pos, keys: keys.clone() });
                }
            }
            s.words.push(text);
        }
    }
    out
}

fn unescape(s: &str) -> String {
    s.replace("&apos;", "'").replace("&quot;", "\"").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
}

fn attr(line: &str, name: &str) -> Option<String> {
    let start = line.find(&format!(" {name}=\""))? + name.len() + 3;
    let len = line[start..].find('"')?;
    Some(unescape(&line[start..start + len]))
}

fn element_text(line: &str) -> Option<String> {
    let start = line.find('>')? + 1;
    let end = line.rfind("</")?;
    (end >= start).then(|| unescape(&line[start..end]))
}

/// Sense key → synset index, from the OEWN `entries-*.json` files (OEWN
/// sense ids are WordNet 3.0-style sense keys).
pub fn sense_keys(oewn_dir: &Path, lex: &Lexicon) -> Result<HashMap<String, u32>, Box<dyn std::error::Error>> {
    let by_id: HashMap<&str, u32> = lex.synsets.iter().enumerate().map(|(i, s)| (s.id.as_str(), i as u32)).collect();
    let mut out = HashMap::new();
    for entry in std::fs::read_dir(oewn_dir)? {
        let path = entry?.path();
        if !path.file_name().is_some_and(|n| n.to_string_lossy().starts_with("entries-")) {
            continue;
        }
        let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&std::fs::read(&path)?)?;
        for by_pos in map.values().filter_map(|v| v.as_object()) {
            for senses in by_pos.values().filter_map(|e| e["sense"].as_array()) {
                for s in senses {
                    if let (Some(key), Some(&synset)) = (s["id"].as_str(), s["synset"].as_str().and_then(|id| by_id.get(id))) {
                        out.insert(key.to_string(), synset);
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<?xml version="1.0" encoding="UTF-8" ?>
<corpus lang="en" source="semcor">
<text id="d000" source="br-e30">
<sentence id="d000.s000">
<wf lemma="how" pos="ADV">How</wf>
<instance id="d000.s000.t000" lemma="long" pos="ADJ">long</instance>
<wf lemma="it" pos="PRON">it</wf>
<instance id="d000.s000.t001" lemma="Rock_&amp;_Roll" pos="NOUN">rock_&amp;_roll</instance>
<wf lemma="'s" pos="PRT">&apos;s</wf>
</sentence>
</text>
<text id="d001" source="br-x">
<sentence id="d001.s000">
<instance id="d001.s000.t000" lemma="run" pos="VERB">ran</instance>
<instance id="d001.s000.t001" lemma="fast" pos="ADV">fast</instance>
</sentence>
</text>
</corpus>"#;

    const GOLD: &str = "d000.s000.t000 long%3:00:02::\nd000.s000.t001 rock_&_roll%1:10:00::\nd001.s000.t000 run%2:38:00:: run%2:38:04::\n";

    #[test]
    fn reads_words_instances_and_documents() {
        let s = parse(XML, GOLD);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].document, "d000");
        assert_eq!(s[0].words, vec!["How", "long", "it", "rock_&_roll", "'s"]);
        assert_eq!(s[0].instances[0], Instance { index: 1, lemma: "long".into(), pos: Pos::Adj, keys: vec!["long%3:00:02::".into()] });
        assert_eq!(s[0].instances[1].lemma, "rock_&_roll");
        assert_eq!(s[1].document, "d001");
    }

    #[test]
    fn keeps_every_gold_key_and_skips_untagged_instances() {
        let s = parse(XML, GOLD);
        assert_eq!(s[1].instances.len(), 1, "fast has no gold key");
        assert_eq!(s[1].instances[0].keys, vec!["run%2:38:00::", "run%2:38:04::"]);
        assert_eq!(s[1].words, vec!["ran", "fast"], "still part of the sentence");
    }
}
