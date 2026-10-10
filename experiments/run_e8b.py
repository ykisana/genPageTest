"""E8b: Jev scenarios on the same 200 test pages as E8 (paired).

  jev                 E8 as run: GenPage rows + top-50 candidates, all rows answered in one parallel call
  jev-dedup           same, but each row gets its own non-overlapping candidate pool (no cross-row collisions)
  jev-seq             one call per row; each call sees what is already on the page (row-level autoregression)
  popularity          cold-start reference: heuristic rows + most-popular eligible titles, no learning
  jev-popular-dedup   cold start: same heuristic rows + popularity candidate pools, Jev ranks them (no GenPage)
  genpage-5%          GenPage trained on 5% of training users (data-scarce)
"""
import argparse
import os
from pathlib import Path

import numpy as np

_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k.startswith("TYPESAFE_") and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")

from common import R, L, catalog, has_ckpt, load_model, run_arm, save, setup, devices  # noqa: E402
from genpage.decoding.baselines import PopularityBaseline  # noqa: E402
from genpage.decoding.page import decode_page  # noqa: E402
from genpage.jev_adapter import CachedTransport, JevDecision, SdkTransport  # noqa: E402


def boot(d, n=10000):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    bs = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--dev", default="mps")
a = ap.parse_args()
if not os.environ.get("TYPESAFE_API_KEY"):
    raise SystemExit("E8b skipped: put TYPESAFE_API_KEY in .env")
dev = devices(a.dev)[0]
tok, imps = setup(1.0, n_test=a.n)
cat = catalog()
m = load_model(1.0, "S", "wbc", dev)
if os.environ.get("JEV_DRY") == "1":   # offline smoke test: rank options in the order given, no API calls
    tr = lambda p: {"answers": {q["id"]: {o["id"]: 1.0 / (i + 1) for i, o in enumerate(q["options"])}
                                for q in p["questions"]}}
else:
    tr = CachedTransport(SdkTransport())

arms = {"genpage_k3": lambda imp, c: decode_page(m, tok, imp["ctx"].tolist(), c, R, L, 3)}
jevs = {
    "jev": JevDecision(tr, m, tok, cat, R, L),
    "jev_dedup": JevDecision(tr, m, tok, cat, R, L, dedup=True),
    "jev_seq": JevDecision(tr, m, tok, cat, R, L, sequential=True),
    "jev_popular_dedup": JevDecision(tr, None, tok, cat, R, L, source="popular", dedup=True),
}
arms.update({k: j.decide_page for k, j in jevs.items()})
arms["popularity"] = PopularityBaseline(tok, cat, R, L, dev).decide_page
if has_ckpt(1.0, "S", "wbc_frac5"):
    m5 = load_model(1.0, "S", "wbc_frac5", dev)
    arms["genpage_5pct"] = lambda imp, c: decode_page(m5, tok, imp["ctx"].tolist(), c, R, L, 3)

res = {}
for name, fn in arms.items():
    res[name], _ = run_arm(name, fn, imps, dev, warmup=0 if name.startswith("jev") else 10)
    if name in jevs:
        res[name]["usage"] = jevs[name].usage

cmp = {}
for ref in ("genpage_k3", "popularity", "genpage_5pct"):
    if ref not in res:
        continue
    for name, r in res.items():
        if name == ref:
            continue
        row = {}
        for met in ("gain", "recall", "diversity"):
            mean, lo, hi = boot(np.array(r["per_page"][met], float) - np.array(res[ref]["per_page"][met], float))
            row[met] = dict(diff=mean, ci95=[lo, hi], significant=bool(lo > 0 or hi < 0))
        cmp[f"{name}_vs_{ref}"] = row
        g = row["gain"]
        print(f"{name:<20} vs {ref:<12} Δgain {g['diff']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]"
              f" {'*' if g['significant'] else 'n.s.'}   Δrecall {row['recall']['diff']:+.4f}"
              f" {'*' if row['recall']['significant'] else 'n.s.'}")
for k, j in jevs.items():
    print(f"{k:<20} tokens in/out {j.usage['input_tokens']:,}/{j.usage['output_tokens']:,} over {j.usage['calls']} calls")
save("e8b_jev_scenarios", dict(arms=res, comparisons=cmp))
