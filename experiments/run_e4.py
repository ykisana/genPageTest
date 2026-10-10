"""E4: MTP heads as a lossless draft (verified) and as a lossy direct decoder."""
import argparse

from common import R, L, load_model, run_arm, save, setup, devices
from genpage.decoding.page import decode_page

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--dev", default="mps")
ap.add_argument("--lam", type=float, default=1.0)
a = ap.parse_args()
dev = devices(a.dev)[0]
tok, imps = setup(a.lam, n_test=a.n)
m = load_model(a.lam, "S", "mtp", dev)
res = {}
for k in (3, 5):
    f = lambda fill: (lambda imp, c: decode_page(m, tok, imp["ctx"].tolist(), c, R, L, k, fill=fill))
    print(f"k={k}")
    res[f"k{k}/ar"], bouts = run_arm("hybrid (AR)", f("ar"), imps, dev)
    for fill in ("spec", "mtp_spec", "mtp_direct"):
        res[f"k{k}/{fill}"], _ = run_arm(fill, f(fill), imps, dev, baseline=bouts)
save("e4_mtp", res)
