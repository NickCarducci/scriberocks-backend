// FidelityAI sense pilot: score each ALL test case once per modeled sense
// (the sense token in the word's slot) with the portable scorer, and pick
// the most typical (lowest score). Compared with first-listed on the same cases.
//
//   node scripts/fidelity_score.mjs <scorer.js> <artifact.json> [data/fidelity]
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const [scorerPath, artifactPath, dir = 'data/fidelity'] = process.argv.slice(2)
const { scoreEncodedSequences } = await import(pathToFileURL(scorerPath).href)
// Accept the artifact itself or the API's /result response, whose artifact is
// result.client_handoff_artifact.
const saved = JSON.parse(readFileSync(artifactPath, 'utf8'))
const artifact = saved.result?.client_handoff_artifact ?? saved.client_handoff_artifact ?? saved
const token = JSON.parse(readFileSync(`${dir}/senses_dense.json`, 'utf8'))
const tests = readFileSync(`${dir}/tests_dense.jsonl`, 'utf8').trim().split('\n').map(l => JSON.parse(l))

let first = 0, fidelity = 0, reachable = 0, errors = 0
const byLemma = {}
for (const t of tests) {
  const modeled = t.senses.filter(s => token[s] != null)
  const seqs = modeled.map(s => t.sequence.map((x, i) => (i === 3 ? token[s] : x)))
  const rows = scoreEncodedSequences(seqs, artifact)
  const scored = modeled.map((s, i) => [s, rows[i].score]).filter(([, x]) => Number.isFinite(x))
  errors += modeled.length - scored.length
  const pick = scored.length ? scored.reduce((a, b) => (b[1] < a[1] ? b : a))[0] : t.first
  const key = `${t.lemma}.${t.pos}`
  byLemma[key] ??= { n: 0, first: 0, fidelity: 0 }
  byLemma[key].n++
  if (t.gold.includes(t.first)) { first++; byLemma[key].first++ }
  if (t.gold.includes(pick)) { fidelity++; byLemma[key].fidelity++ }
  if (t.gold.some(g => modeled.includes(g))) reachable++
}
const pct = x => `${(100 * x / tests.length).toFixed(1)}%`
console.log(`${tests.length} test cases: first-listed ${pct(first)}, FidelityAI ${pct(fidelity)} (gold among modeled senses ${pct(reachable)}; ${errors} scoring errors)`)
for (const [k, v] of Object.entries(byLemma)) console.log(`  ${k.padEnd(12)} ${String(v.n).padStart(3)}  first-listed ${v.first}  FidelityAI ${v.fidelity}`)
