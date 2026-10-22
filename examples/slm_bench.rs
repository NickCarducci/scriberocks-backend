//! Times one training step of the flagship's shape (30,000-piece vocabulary,
//! 4 × 256 pre-LN, batch 128) on the Metal GPU, to find where step time goes.
//! SCRIBE_BENCH_FORWARD_ONLY / SCRIBE_BENCH_NO_STEP split forward, backward
//! and optimizer; SCRIBE_BENCH_VOCAB sets the vocabulary size.
//! Findings (Oct 22 2026, M5 Pro, 128 × 26 pieces): forward 38 ms, backward
//! 154 ms, AdamW 4 ms. Hand-written LayerNorm ≈ 25% of a step; vocabulary-sized
//! backward ≈ 45% (192 ms at 30,000 pieces vs 103 at 2,000), not the embedding
//! or gather scatters: likely candle's Metal matmul backward for the 30,000 × 256
//! output layer. Fused forward ops alone save ≈ 10%.
//!   cargo run --release --features metal --example slm_bench [pieces per sentence]
use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use scribe::bert::Bert;
use scribe::slm_train::{masked_loss, Rng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let len: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(24);
    let dev = Device::new_metal(0)?;
    let dir = std::env::temp_dir().join("scribe-bench");
    std::fs::create_dir_all(&dir)?;
    let vsize: usize = std::env::var("SCRIBE_BENCH_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(30_000);
    let vocab: Vec<String> = (0..vsize).map(|i| match i { 0 => "[PAD]".into(), 1 => "[UNK]".into(), 2 => "[CLS]".into(), 3 => "[SEP]".into(), 4 => "[MASK]".into(), _ => format!("w{i}") }).collect();
    std::fs::write(dir.join("vocab.txt"), vocab.join("\n"))?;
    std::fs::write(dir.join("config.json"), serde_json::json!({"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4,
        "intermediate_size": 1024, "max_position_embeddings": 128, "layer_norm_eps": 1e-12, "pre_layer_norm": true}).to_string())?;
    let (bert, vars) = Bert::new_random(&dir, dev.clone())?;
    let mut opt = AdamW::new(vars.all_vars(), ParamsAdamW { lr: 1e-4, ..Default::default() })?;
    let mut rng = Rng(1);
    let rows: Vec<Vec<u32>> = (0..128).map(|_| { let mut r = vec![2u32]; r.extend((0..len).map(|_| 5 + (rng.next() % (vsize as u64 - 5)) as u32)); r.push(3); r }).collect();
    let at: Vec<(usize, usize)> = (0..128).flat_map(|r| (1..=len).filter(move |c| c % 7 == 0).map(move |c| (r, c))).collect();
    let gold: Vec<u32> = at.iter().map(|&(r, c)| rows[r][c]).collect();
    let forward_only = std::env::var_os("SCRIBE_BENCH_FORWARD_ONLY").is_some();
    let mut step = || -> candle_core::Result<()> {
        let l = masked_loss(&bert, &rows, &at, &gold, &dev)?;
        if forward_only { return l.to_scalar::<f32>().map(|_| ()) }
        if std::env::var_os("SCRIBE_BENCH_NO_STEP").is_some() {
            let g = l.backward()?;
            return g.get(&vars.all_vars()[0]).map(|t| t.sum_all().and_then(|x| x.to_scalar::<f32>()).map(|_| ())).unwrap_or(Ok(()));
        }
        opt.backward_step(&l)
    };
    for _ in 0..3 { step()? }
    dev.synchronize()?;
    let t = std::time::Instant::now();
    let n = 15;
    for _ in 0..n { step()? }
    dev.synchronize()?;
    println!("{:.0} ms per step (batch 128 × {} pieces, {} masked)", t.elapsed().as_secs_f64() * 1000.0 / n as f64, len + 2, at.len());
    Ok(())
}
