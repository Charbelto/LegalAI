# Archived: pre-fix benchmark artifacts (2026-07-25)

These files are kept for provenance only. **Do not cite any number in them.**

| File | Why it is invalid |
|---|---|
| `benchmark_results.json` | Oldest run: 1 query x 1 repeat x 6 modes. Every figure in the paper drafts (79.07 s vs 46.51 s, "~40% faster", all BLEU/ROUGE values) comes from this single unrepeated observation. |
| `benchmark_runs.jsonl` | 324 rows, 9 queries (all of type `simple`) x 8 modes x 5 repeats. Ruined by the aggregator abstention veto: 7 of 8 topologies returned the 7-word abstention string. Also polluted by the `compl_ai` canned-answer hijack, and smoke rows are mixed into repeat 0. |
| `analysis_summary.csv`, `by_query_type.csv`, `significance.csv`, `results.json`, `metrics_table.tex`, `results/` | Produced from a degenerate smoke run (n = 1 per mode, n = 4 for `all`). All ROUGE = 0.0, all p-values = 1.0, all CIs +/- 0.00. `run_meta.json` recorded a hardcoded `duration_s: 300.0`. |

The defects behind these artifacts are fixed in the code as of 2026-07-25; see
`../../FIXES_APPLIED.md` in the project root. A clean re-run writes fresh files
to the normal locations.
