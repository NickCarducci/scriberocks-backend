//! Compresses a BERT-format model folder for shipping in the app.
//!
//!   cargo run --release --example compress_model <in dir> <out dir> f16|q8
//!
//! f16: half precision (~half the size). q8: 8-bit rows with a scale each
//! (~a quarter; bert::quantize_q8). Both load as full precision.
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (src, dst, kind) = (Path::new(&a[0]), Path::new(&a[1]), a[2].as_str());
    std::fs::create_dir_all(dst)?;
    for f in ["vocab.txt", "config.json"] { std::fs::copy(src.join(f), dst.join(f))?; }
    let w = candle_core::safetensors::load(src.join("model.safetensors"), &candle_core::Device::Cpu)?;
    // Only what inference needs: the BERT encoder and the masked-word head.
    let w: HashMap<String, candle_core::Tensor> = w.into_iter().filter(|(k, _)| !k.starts_with("cls.seq_relationship") && !k.contains("pooler")).collect();
    match kind {
        "f16" => candle_core::safetensors::save(&w.iter().map(|(k, v)| Ok((k.clone(), v.to_dtype(candle_core::DType::F16)?))).collect::<candle_core::Result<HashMap<_, _>>>()?, dst.join("model.safetensors"))?,
        "q8" => candle_core::safetensors::save(&scribe::bert::quantize_q8(&w)?, dst.join("model.q8.safetensors"))?,
        _ => return Err("kind must be f16 or q8".into()),
    }
    println!("{} -> {} ({kind})", src.display(), dst.display());
    Ok(())
}
