"""E8: the real Jev API (TypeSafe System One) as the decision model. Needs TYPESAFE_API_KEY in .env.

Sends public MovieLens titles/ratings (no PII) to TypeSafe's service. Responses are cached in
results/jev_cache/ so reruns don't re-bill. Latency is network-dominated: reported separately.
"""
import argparse
import os
from pathlib import Path

from common import R, L, catalog, load_model, run_arm, save, setup, devices
from genpage.decoding.page import decode_page
from genpage.jev_adapter import CachedTransport, JevDecision, SdkTransport

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--dev", default="mps")
a = ap.parse_args()
# load TYPESAFE_* from a local, gitignored .env (KEY=value lines) without printing anything
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k.startswith("TYPESAFE_") and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")
if not os.environ.get("TYPESAFE_API_KEY"):
    raise SystemExit("E8 skipped: put TYPESAFE_API_KEY in .env (see console.typesafe.ai/settings/keys).")
dev = devices(a.dev)[0]
tok, imps = setup(1.0, n_test=a.n)
m = load_model(1.0, "S", "wbc", dev)
jev = JevDecision(CachedTransport(SdkTransport()), m, tok, catalog(), R, L)
res = {}
res["hybrid_k3"], bouts = run_arm("GenPage hybrid k=3", lambda imp, c: decode_page(m, tok, imp["ctx"].tolist(), c, R, L, 3), imps, dev)
res["jev"], _ = run_arm("Jev (rows+candidates from GenPage)", jev.decide_page, imps, dev, baseline=bouts, warmup=0)
save("e8_jev", res)
