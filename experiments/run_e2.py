"""E2: where does batch-1 serving time go? prefill vs decode forwards vs masks vs selection."""
import argparse
import time

import numpy as np

from common import R, L, catalog, load_model, save, setup, devices
from genpage.decoding.constraints import CatalogTensors, Constraints
from genpage.decoding.page import decode_page
from genpage.eval.profile import Profiler, sync

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--devs", default="mps,cpu")
ap.add_argument("--caps", default="50,100,200,500")
ap.add_argument("--sizes", default="S,M")
a = ap.parse_args()
cat = catalog()
tok, imps = setup(1.0, hist_cap=500, n_test=10**9)
imps = [i for i in imps if (len(i["ctx"]) - 10) // 3 >= 500][: a.n]   # users with >= 500 actions of history
print(f"{len(imps)} impressions with >=500 history")


def cap_ctx(ctx, n):
    head, hist = ctx[:9], ctx[9:-1]
    return np.concatenate([head, hist[len(hist) - 3 * n:], ctx[-1:]]).tolist()


rows = []
for dev in devices(a.devs):
    ct = CatalogTensors(cat, dev)
    for size in a.sizes.split(","):
        m = load_model(1.0, size, "wbc", dev, max_len=2048)
        for cap in [int(c) for c in a.caps.split(",")]:
            for fill, k in [("ar", 1), ("ar", 3), ("ar", 10), ("spec", 3), ("spec", 10)]:
                for _ in range(3):  # warmup
                    decode_page(m, tok, cap_ctx(imps[0]["ctx"], cap), Constraints(ct, imps[0]["watched"], L), R, L, k, fill)
                prof = Profiler(dev)
                e2e = []
                for imp in imps:
                    ctx = cap_ctx(imp["ctx"], cap)
                    decode_page(m, tok, ctx, Constraints(ct, imp["watched"], L), R, L, k, fill, prof=prof)
                    sync(dev)
                    t = time.perf_counter()
                    out = decode_page(m, tok, ctx, Constraints(ct, imp["watched"], L), R, L, k, fill)
                    sync(dev)
                    e2e.append((time.perf_counter() - t) * 1000)
                n = len(imps)
                ph = {k_: v / n * 1000 for k_, v in prof.t.items()}
                row = dict(dev=dev.type, size=size, hist_cap=cap, ctx_len=len(cap_ctx(imps[0]["ctx"], cap)),
                           fill=fill, k=k, forwards=out.forwards, e2e_ms=float(np.mean(e2e)),
                           **{f"{p}_ms": v for p, v in ph.items()})
                tot = sum(ph.values())
                row["prefill_share"] = ph.get("prefill", 0) / tot
                row["mask_share"] = ph.get("mask", 0) / tot
                rows.append(row)
                print(f"{dev.type} {size} cap={cap:<3} {fill:<4} k={k:<2} fwd={out.forwards:<3} e2e={row['e2e_ms']:6.1f}ms "
                      + " ".join(f"{p}={v:.1f}" for p, v in ph.items()))
save("e2_profile", rows)
