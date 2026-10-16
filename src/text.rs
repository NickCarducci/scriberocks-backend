//! Sentence splitting, tokenization (UD/Penn conventions) and detokenization.

/// Splits into sentences on terminal punctuation and line breaks.
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '\n' {
            if !current.trim().is_empty() {
                out.push(std::mem::take(&mut current));
            }
            current.clear();
            continue;
        }
        current.push(c);
        let next = chars.get(i + 1).copied();
        if matches!(c, '.' | '!' | '?') && next.is_none_or(|n| n.is_whitespace()) {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

/// Penn/UD-style tokens: punctuation split off, contractions split ("don't" → "do", "n't").
pub fn tokenize(sentence: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in sentence.split_whitespace() {
        let raw = raw.replace(['’', '‘'], "'");
        let mut word = raw.as_str();
        let mut trailing = Vec::new();
        while let Some(c) = word.chars().next().filter(|c| "\"'([{".contains(*c)) {
            out.push(c.to_string());
            word = &word[c.len_utf8()..];
        }
        while let Some(c) = word.chars().last().filter(|c| ".,!?;:\")]}'".contains(*c)) {
            trailing.push(c.to_string());
            word = &word[..word.len() - c.len_utf8()];
        }
        if !word.is_empty() {
            let lower = word.to_lowercase();
            if let Some(stem) = lower.strip_suffix("n't").filter(|s| !s.is_empty()) {
                out.push(word[..stem.len()].to_string());
                out.push(word[stem.len()..].to_string());
            } else if let Some(pos) = ["'m", "'re", "'s", "'ve", "'ll", "'d"].iter().find_map(|s| lower.strip_suffix(s).filter(|x| !x.is_empty()).map(|x| x.len())) {
                out.push(word[..pos].to_string());
                out.push(word[pos..].to_string());
            } else {
                out.push(word.to_string());
            }
        }
        out.extend(trailing.into_iter().rev());
    }
    out
}

/// Joins tokens back into text: no space before punctuation or clitics
/// ("ca" + "n't" → "can't"), first letter capitalized.
pub fn detokenize<S: AsRef<str>>(tokens: &[S]) -> String {
    let mut s = String::new();
    for (i, t) in tokens.iter().enumerate() {
        let w = t.as_ref();
        let lower = w.to_lowercase();
        let clitic = matches!(lower.as_str(), "n't" | "'s" | "'m" | "'re" | "'ve" | "'ll" | "'d");
        let glue = clitic || matches!(w, "." | "," | "!" | "?" | ";" | ":" | ")" | "”" | "...");
        let after_open = i > 0 && matches!(tokens[i - 1].as_ref(), "(" | "“");
        if !s.is_empty() && !glue && !after_open {
            s.push(' ');
        }
        s.push_str(w);
    }
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().chain(chars).collect(),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_contractions_and_punctuation() {
        assert_eq!(tokenize("I can't go, I'm tired."), ["I", "ca", "n't", "go", ",", "I", "'m", "tired", "."]);
        assert_eq!(tokenize("Don’t."), ["Do", "n't", "."]);
    }

    #[test]
    fn round_trips_through_detokenize() {
        for s in ["I can't go, I'm tired.", "We won't stay.", "Why am I always afraid to ask for help?"] {
            assert_eq!(detokenize(&tokenize(s)), s);
        }
    }

    #[test]
    fn splits_sentences_on_terminals_and_newlines() {
        assert_eq!(sentences("One. Two?\nThree"), ["One.", " Two?", "Three"]);
        assert_eq!(sentences("v1.6 shipped"), ["v1.6 shipped"]);
    }
}
