"""E1: baseline mini-GenPage with hybrid row decoding, vs. the logged teacher pages; Table-1 analogue."""
import argparse
import json

from common import R, L, RESULTS_DIR, load_model, run_arm, save, setup, teacher_pages, devices
from genpage.decoding.page import decode_page

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--dev", default="mps")
ap.add_argument("--lams", default="0,1,2")
a = ap.parse_args()
dev = devices(a.dev)[0]
res = {"arms": {}, "table1": {}}
for lam in [float(x) for x in a.lams.split(",")]:
    tok, imps = setup(lam, n_test=a.n)
    print(f"lambda={lam:g}")
    tp = teacher_pages(imps)
    res["arms"][f"lam{lam:g}/teacher"], _ = run_arm("teacher (logged)", lambda imp, c, it=iter(tp): next(it),
                                                    imps, dev, warmup=0, time_it=False)
    arms = [("S", "pretrain"), ("S", "wbc")] + ([("S", "wbc_scratch"), ("M", "wbc")] if lam == 1 else [])
    for size, stage in arms:
        try:
            m = load_model(lam, size, stage, dev)
        except FileNotFoundError:
            print(f"  (missing {size}-{stage})")
            continue
        fn = lambda imp, c, m=m: decode_page(m, tok, imp["ctx"].tolist(), c, R, L, k=3, fill="ar")
        res["arms"][f"lam{lam:g}/{size}-{stage}"], _ = run_arm(f"{size}-{stage} hybrid k=3", fn, imps, dev)
for stage in ("wbc", "wbc_scratch"):
    p = RESULTS_DIR / f"train_lam1_h100_S_{stage}.json"
    if p.exists():
        res["table1"][stage] = json.loads(p.read_text())["test"]
print("Table 1 analogue:", json.dumps(res["table1"], indent=1))
save("e1_baseline", res)
