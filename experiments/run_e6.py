"""E6: confidence-gated hybrid. A calibrated yes/no decision model chooses k per row."""
import argparse

import numpy as np

from common import R, L, catalog, load_model, run_arm, save, setup, devices
from genpage.decoding.constraints import CatalogTensors, Constraints
from genpage.decoding.gate import Gate, collect
from genpage.decoding.page import decode_page
from genpage.eval.calibration import ece, reliability

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--n-val", type=int, default=400)
ap.add_argument("--dev", default="mps")
ap.add_argument("--lams", default="0,1,2")
ap.add_argument("--kmax", type=int, default=5)
a = ap.parse_args()
dev = devices(a.dev)[0]
ct = CatalogTensors(catalog(), dev)
mk = lambda imp: Constraints(ct, imp["watched"], L)
res = {}
for lam in [float(x) for x in a.lams.split(",")]:
    tok, val = setup(lam, n_test=a.n_val, split="val")
    _, imps = setup(lam, n_test=a.n)
    m = load_model(lam, "S", "wbc", dev)
    X, y = collect(m, tok, val, mk, R, L, a.kmax)
    gate = Gate.fit(X, y, dev)
    Xt, yt = collect(m, tok, imps, mk, R, L, a.kmax)
    pt = gate.prob_np(Xt)
    r = dict(gate=dict(n_train=len(y), base_rate_train=float(y.mean()), base_rate_test=float(yt.mean()),
                       ece=ece(pt, yt), reliability=reliability(pt, yt),
                       accuracy=float(((pt >= 0.5) == yt).mean())))
    print(f"lambda={lam:g}: P(stop is lossless) {yt.mean():.3f}, ECE {r['gate']['ece']:.4f}")
    f = lambda **kw: (lambda imp, c: decode_page(m, tok, imp["ctx"].tolist(), c, R, L, **kw))
    r[f"ar_k{a.kmax}"], bouts = run_arm(f"hybrid k={a.kmax}", f(k=a.kmax), imps, dev)
    for k in range(1, a.kmax):
        r[f"ar_k{k}"], _ = run_arm(f"hybrid k={k}", f(k=k), imps, dev, baseline=bouts)
    for th in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        r[f"gated_{th}"], _ = run_arm(f"gated th={th}", f(k=a.kmax, fill="gated", gate=gate, gate_threshold=th),
                                      imps, dev, baseline=bouts)
    r["gated_oracle"], _ = run_arm("gated oracle", f(k=a.kmax, fill="oracle"), imps, dev, baseline=bouts)
    res[f"lam{lam:g}"] = r
save("e6_gated", res)
