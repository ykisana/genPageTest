**Note**: The following experiments were written by Claude (with human guidance). Pretty much everything below (except this note section, was written by AI).

This repo is just an experiment to mimic and play around with Netflix's GenPage concepts. I wanted to see if I could replicate it, and detects bottlenecks. While look at ways to speed it up.

For speeding it up, I had a few ideas. First, playing around initial one-by-one generation per row, we call this (k). The paper said 'first few entries' so I tried to find optimal first few. I'm assuming Netflix already played around with this, but was interesting to see how results change. 

More interestingly, I wanted to look at **skeleton first**, changing the order of generation. Rather than just going one-by-one, what if we outline the skeleton, then fill the rows based on that. For example, the page may order (comedies, action, trending, for you) in that order. Then fill out each row. Still following the LLM-style fill, with one distinction **the rows are filled at the same time**. Within each row it's still one-by-one like GenPage. We saw a **3.8% gain in page quality and 1.9x speed compared against a GenPage baseline with the same training budget**. Which is HUGE, but of-course at a much smaller scale with different data.

Second I wanted to see how **Decision/System One** models, can speed things up. Such as replacing GenPage, where every slot had questions it would go through before filling. And others like checking with it when it should stop one-by-one generation.

As it explains below, this was done on a much, much smaller scale, so take these results with a grain of salt. 

Results are [here](https://github.com/ykisana/genPageTest/blob/main/results/report.md).

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
