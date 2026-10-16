use scribe::{wsd, Lexicon};
use std::path::Path;
fn main() {
    let lex = Lexicon::load(Path::new("models/lexicon.bin")).unwrap();
    let (n, fine, first, coarse, first_coarse) = wsd::evaluate(&lex, 1.0);
    println!("{n} held-out dictionary examples");
    println!("  exact sense:      overlap {fine:.1}%   first-listed {first:.1}%");
    println!("  right category:   overlap {coarse:.1}%   first-listed {first_coarse:.1}%");
}
