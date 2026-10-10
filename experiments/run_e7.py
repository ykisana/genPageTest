"""E7: Jev-style fully parallel decision model replaces the decoder; compared to GenPage hybrid."""
import argparse

from common import R, L, has_ckpt, load_model, run_arm, save, setup, devices
from genpage.decision import LocalParallelDecision
from genpage.decoding.page import decode_page

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--devs", default="mps,cpu")
ap.add_argument("--lams", default="0,1,2")
a = ap.parse_args()
res = []
for dev in devices(a.devs):
    for lam in [float(x) for x in a.lams.split(",")]:
        tok, imps = setup(lam, n_test=a.n)
        hyb = load_model(lam, "S", "wbc", dev)
        print(f"{dev.type} lambda={lam:g}")
        row = dict(dev=dev.type, lam=lam)
        row["hybrid_k3"], bouts = run_arm("GenPage hybrid k=3", lambda imp, c: decode_page(hyb, tok, imp["ctx"].tolist(), c, R, L, 3), imps, dev)
        row["hybrid_k10"], _ = run_arm("GenPage full AR (k=10)", lambda imp, c: decode_page(hyb, tok, imp["ctx"].tolist(), c, R, L, 10), imps, dev, baseline=bouts)
        for arm in ("parallel_indep", "parallel_joint"):
            if not has_ckpt(lam, "S", arm):
                continue
            dm = LocalParallelDecision(load_model(lam, "S", arm, dev), tok, R, L, joint=arm.endswith("joint"))
            row[arm], _ = run_arm(dm.name, dm.decide_page, imps, dev, baseline=bouts)
        res.append(row)
save("e7_parallel", res)
