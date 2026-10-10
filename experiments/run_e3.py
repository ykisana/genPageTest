"""E3: self-speculative hybrid decoding: acceptance, exactness vs. AR, forwards, latency."""
import argparse

from common import R, L, load_model, run_arm, save, setup, devices
from genpage.decoding.page import decode_page

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--devs", default="mps,cpu")
ap.add_argument("--lams", default="0,1,2")
ap.add_argument("--ks", default="3,5,10")
a = ap.parse_args()
res = []
for dev in devices(a.devs):
    for lam in [float(x) for x in a.lams.split(",")]:
        tok, imps = setup(lam, n_test=a.n)
        arms = [("S", "wbc"), ("S", "pretrain")] + ([("M", "wbc")] if lam == 1 else [])
        for size, stage in arms:
            try:
                m = load_model(lam, size, stage, dev)
            except FileNotFoundError:
                continue
            for k in [int(x) for x in a.ks.split(",")]:
                if stage == "pretrain" and k != 3:
                    continue
                print(f"{dev.type} lambda={lam:g} {size}-{stage} k={k}")
                f = lambda fill: (lambda imp, c: decode_page(m, tok, imp["ctx"].tolist(), c, R, L, k, fill=fill))
                base, bouts = run_arm("hybrid (AR)", f("ar"), imps, dev)
                spec, _ = run_arm("self-spec", f("spec"), imps, dev, baseline=bouts)
                res.append(dict(dev=dev.type, lam=lam, size=size, stage=stage, k=k, ar=base, spec=spec,
                                speedup=base["latency_ms_mean"] / spec["latency_ms_mean"],
                                fwd_ratio=base["forwards"] / spec["forwards"]))
save("e3_selfspec", res)
