# mini-GenPage: decoding-latency experiments

A small, open reproduction of the serving side of **GenPage** (Wang, Pan, Baltrunas — *GenPage: Towards
End-to-End Generative Homepage Construction at Netflix*, RecSys '26, arXiv 2606.31031) on MovieLens-1M,
used to measure:

| # | Experiment | Changes pages? |
|---|---|---|
| E1 | Baseline: custom tokenizer, NTP pretraining → WBC post-training, hybrid row decoding, constrained decoding | – |
| E2 | Where batch-1 serving time goes (prefill / decode forwards / masks / selection) vs. history length and model size | – |
| E3 | **Self-speculative decoding**: the masked ranking already computed at each step is the draft; verify k entities in one forward | no (lossless) |
| E4 | Medusa-style MTP heads, as a verified draft (lossless) and unverified (lossy) | (b) yes |
| E5 | Skeleton + parallel fill (all row tokens first, then rows filled as a batch) | yes |
| E6 | **Jev-style decision model as a component**: a calibrated yes/no gate that picks k per row | yes |
| E7 | **Jev-style decision model as the decoder**: one parallel pass answering every slot as a declared choice | yes |
| E8 | The real TypeSafe Jev API behind the same interface | yes |
| E8b | Jev scenarios: deduplicated candidates, row-by-row calls, cold start (vs popularity and a 5%-data GenPage) | yes |
| – | Paired-bootstrap significance on 1000 test pages per λ, plus an equal-training-budget control for the skeleton | – |

Results: [`results/report.md`](results/report.md) (narrative) and `results/report_tables.md` (all numbers), plots in `plots/`.

## What maps to what

| GenPage (paper) | here |
|---|---|
| Context tokens: actions (entity, action, time, duration), profile, request context, segment markers | `<HIST>` (movie, rating-bucket, Δt-bucket)×N, `<PROFILE>` gender/age/occupation, `<REQ>` hour/weekday |
| Page tokens: one token per row and per entity, layout order | 6 rows × 10 titles; 26 row types (genres, decades, Top Picks, Popular, Highly Rated) |
| Production pages + reward system | a **synthetic teacher** (SVD fold-in + MMR) logs pages; reward from the user's next 30 ratings |
| Pretraining on positively engaged pages, then WBC (sign → label, \|reward\| → weight, random negatives), untied embeddings | same |
| Constrained decoding with per-request masks | Top Picks pinned at row 0, no repeated rows/titles, eligibility per row, no watched titles |
| Hybrid row decoding | first k entities autoregressive, tail from one forward pass |

**The teacher's λ is the controlled variable.** Within a row the teacher picks titles by MMR with strength λ, so
λ=0 means a title's value doesn't depend on the titles already in the row and λ=2 means it depends strongly.
Teacher quality (hits/page) is flat across λ. This makes "how much does autoregression matter?" measurable:
it is exactly what speculative acceptance and the parallel models are sensitive to.

## Reproduce

```bash
python3 -m venv .venv --system-site-packages && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m genpage.data.download
for l in 0 1 2; do .venv/bin/python -m genpage.data.build --lam $l; done
.venv/bin/python -m genpage.data.build --lam 1 --hist-cap 500     # long-history contexts for E2
./experiments/train_all.sh          # every checkpoint, resumable (hours on an Apple M3)
./experiments/run_experiments.sh   # E1–E8 + report tables/plots
./experiments/followups3.sh        # 5%-data GenPage, significance (1000 pages/λ), Jev scenarios (E8b)
.venv/bin/python -m pytest tests    # tokenizer, KV cache, masks, losslessness, layouts, Jev adapter
```

Smoke test everything on 300 users: `--tiny` on `genpage.data.build` / `genpage.train.run` / `genpage.train.arms`,
then `GENPAGE_TINY=1 python experiments/run_eN.py --n 15 --devs cpu`.

E8 (real Jev): `genpage/jev_adapter.py` calls TypeSafe's System One through the official `typesafe-sdk`
(one call per page, one `Choice` per row over GenPage's top-50 eligible candidates; Jev's per-option
probabilities rank the row). Put `TYPESAFE_API_KEY=...` in a local `.env` (gitignored); responses are cached in
`results/jev_cache/` so reruns don't re-bill. It sends public MovieLens titles and ratings, no PII.

## Not reproduced

RL post-training (Dr. GRPO), pagination with in-session signals, semantic-embedding fusion / cold start,
multi-cadence incremental training, and anything at Netflix scale (the S model is ~3M non-embedding params,
not 200M).
