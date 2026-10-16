//! Minimal CoNLL-U reader for Universal Dependencies treebanks.

#[derive(Clone, Debug)]
pub struct Token {
    pub form: String,
    pub lemma: String,
    pub upos: String,
    /// Penn Treebank tag (tense, number, degree).
    pub xpos: String,
    /// 1-based index of the head word (0 = root).
    pub head: usize,
    pub deprel: String,
}

pub type Sentence = Vec<Token>;

pub fn parse(text: &str) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    let mut current = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                sentences.push(std::mem::take(&mut current));
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        // Skip multiword ranges ("1-2") and empty nodes ("8.1").
        if cols.len() < 8 || cols[0].contains('-') || cols[0].contains('.') {
            continue;
        }
        current.push(Token {
            form: cols[1].to_string(),
            lemma: cols[2].to_string(),
            upos: cols[3].to_string(),
            xpos: cols[4].to_string(),
            head: cols[6].parse().unwrap_or(0),
            deprel: cols[7].to_string(),
        });
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}
