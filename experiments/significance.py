"""Paired bootstrap CIs for the key page-quality comparisons (same test pages, same order).

Re-runs each comparison with per-page metrics on the test set and reports mean difference in
discounted gain with a 95% CI, plus the latency ratio.
"""
import argparse

import numpy as np

from common import R, L, has_ckpt, load_model, run_arm, save, setup, devices, teacher_pages
from genpage.decision import LocalParallelDecision
from genpage.decoding.page import decode_page
from genpage.decoding.parallel_fill import decode_skeleton


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--dev", default="mps")
ap.add_argument("--lam", type=float, default=1.0)
a = ap.parse_args()
dev = devices(a.dev)[0]
tok, imps = setup(a.lam, n_test=a.n)
ar = lambda m, k, fill="ar": (lambda imp, c: decode_page(m, tok, imp["ctx"].tolist(), c, R, L, k, fill=fill))
wbc = load_model(a.lam, "S", "wbc", dev)
arms = {"teacher": (lambda imp, c, it=iter(teacher_pages(imps)): next(it), False)}
arms["hybrid_k3"] = (ar(wbc, 3), True)
arms["hybrid_k1"] = (ar(wbc, 1), True)
arms["hybrid_k10"] = (ar(wbc, 10), True)
if has_ckpt(a.lam, "S", "mtp"):
    mtp = load_model(a.lam, "S", "mtp", dev)
    arms["mtp_direct_k3"] = (ar(mtp, 3, "mtp_direct"), True)
if has_ckpt(a.lam, "S", "skeleton"):
    sk = load_model(a.lam, "S", "skeleton", dev)
    arms["skeleton_k3"] = (lambda imp, c: decode_skeleton(sk, tok, imp["ctx"].tolist(), c, R, L, 3), True)
for tag in ("_ctrl3",):
    if has_ckpt(a.lam, "S", "wbc" + tag):
        arms["hybrid_k3" + tag] = (ar(load_model(a.lam, "S", "wbc" + tag, dev), 3), True)
for arm in ("parallel_indep", "parallel_joint"):
    if has_ckpt(a.lam, "S", arm):
        dm = LocalParallelDecision(load_model(a.lam, "S", arm, dev), tok, R, L, joint=arm.endswith("joint"))
        arms[arm] = (dm.decide_page, True)

res = {}
for name, (fn, timed) in arms.items():
    res[name], _ = run_arm(name, fn, imps, dev, warmup=10 if timed else 0, time_it=timed)
base = res["hybrid_k3"]["per_page"]
out = {"n": len(imps), "lam": a.lam, "vs_hybrid_k3": {}}
for name, r in res.items():
    if name == "hybrid_k3":
        continue
    d = np.array(r["per_page"]["gain"]) - np.array(base["gain"])
    mean, lo, hi = boot_ci(d)
    row = dict(gain=r["gain"], d_gain=mean, ci95=[lo, hi], rel=mean / res["hybrid_k3"]["gain"],
               d_recall=float(np.mean(np.array(r["per_page"]["recall"], float) - np.array(base["recall"], float))),
               d_diversity=float(np.mean(np.array(r["per_page"]["diversity"]) - np.array(base["diversity"]))))
    if "latency_ms" in r["per_page"]:
        row["latency_ratio"] = res["hybrid_k3"]["latency_ms_mean"] / r["latency_ms_mean"]
        row["fwd"] = r["forwards"]
    out["vs_hybrid_k3"][name] = row
    sig = "significant" if lo > 0 or hi < 0 else "n.s."
    print(f"{name:<22} Δgain {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] ({row['rel']:+.1%}) {sig}"
          + (f"  speed {row['latency_ratio']:.2f}x" if "latency_ratio" in row else ""))
save(f"significance_lam{a.lam:g}", out)
