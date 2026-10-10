"""E5: skeleton + parallel fill vs. hybrid row decoding (changes outputs)."""
import argparse

from common import R, L, has_ckpt, load_model, run_arm, save, setup, devices
from genpage.decoding.page import decode_page
from genpage.decoding.parallel_fill import decode_skeleton

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--devs", default="mps,cpu")
ap.add_argument("--lams", default="0,1,2")
a = ap.parse_args()
res = []
for dev in devices(a.devs):
    for lam in [float(x) for x in a.lams.split(",")]:
        if not has_ckpt(lam, "S", "skeleton"):
            continue
        tok, imps = setup(lam, n_test=a.n)
        hyb = load_model(lam, "S", "wbc", dev)
        sk = load_model(lam, "S", "skeleton", dev)
        print(f"{dev.type} lambda={lam:g}")
        for k in (1, 3):
            base, bouts = run_arm(f"hybrid k={k}", lambda imp, c: decode_page(hyb, tok, imp["ctx"].tolist(), c, R, L, k), imps, dev)
            skel, _ = run_arm(f"skeleton k={k}", lambda imp, c: decode_skeleton(sk, tok, imp["ctx"].tolist(), c, R, L, k), imps, dev, baseline=bouts)
            res.append(dict(dev=dev.type, lam=lam, k=k, hybrid=base, skeleton=skel))
save("e5_skeleton", res)
