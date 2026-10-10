"""Shared harness: run a decoder over held-out impressions and collect quality + latency + stats."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genpage.config import PLOTS_DIR, RESULTS_DIR, DataConfig, ModelConfig  # noqa: E402
from genpage.data.catalog import load_catalog  # noqa: E402
from genpage.decoding.constraints import CatalogTensors, Constraints  # noqa: E402
from genpage.eval.calibration import ece  # noqa: E402
from genpage.eval.metrics import agreement, page_metrics, violations  # noqa: E402
from genpage.eval.profile import sync  # noqa: E402
from genpage.model import GenPageLM  # noqa: E402
from genpage.train.core import ckpt_path, load_split  # noqa: E402

R, L = 6, 10
TINY = os.environ.get("GENPAGE_TINY") == "1"   # smoke-test against the --tiny datasets/checkpoints
_CAT = None


def catalog():
    global _CAT
    if _CAT is None:
        _CAT = load_catalog()
    return _CAT


def setup(lam=1.0, hist_cap=100, n_test=500, split="test", seed=0):
    tok, by = load_split(DataConfig(mmr_lambda=lam, hist_cap=hist_cap), tiny=TINY)
    imps = by[split]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imps), size=min(n_test, len(imps)), replace=False)
    return tok, [imps[i] for i in sorted(idx)]


def load_model(lam, size, stage, dev, max_len=None):
    p = ckpt_path(DataConfig(mmr_lambda=lam), size, stage, tiny=TINY)
    d = torch.load(p, map_location="cpu", weights_only=False)
    cfg = d["cfg"] if max_len is None else {**d["cfg"], "max_len": max_len}
    m = GenPageLM(ModelConfig(**cfg))
    m.load_state_dict(d["state"])
    return m.to(dev).eval()


def has_ckpt(lam, size, stage):
    return ckpt_path(DataConfig(mmr_lambda=lam), size, stage, tiny=TINY).exists()


def run_arm(name, decode_fn, imps, dev, baseline=None, warmup=10, time_it=True):
    """decode_fn(imp, cons) -> PageOut. Returns (summary dict, list of PageOut)."""
    cat = catalog()
    ct = CatalogTensors(cat, dev)
    mk = lambda imp: Constraints(ct, imp["watched"], L)
    for imp in imps[:warmup]:
        decode_fn(imp, mk(imp))
    outs, recs, lat = [], [], []
    conf, lab = [], []
    for i, imp in enumerate(imps):
        cons = mk(imp)
        sync(dev)
        t0 = time.perf_counter()
        out = decode_fn(imp, cons)
        sync(dev)
        lat.append((time.perf_counter() - t0) * 1000)
        outs.append(out)
        m = page_metrics(out.rows, out.ents, imp, cat)
        m["violations"] = violations(out.rows, out.ents, imp, cat, L)
        m["forwards"] = out.forwards
        if baseline is not None:
            m.update(agreement(out.ents, baseline[i].ents))
            m["rows_agree"] = float(out.rows == baseline[i].rows)
        recs.append(m)
        fut = dict(zip(imp["fut_items"].tolist(), imp["fut_ratings"].tolist()))
        for es, vs in zip(out.ents, out.values):
            for e, v in zip(es, vs):
                conf.append(1 / (1 + np.exp(-v)))
                lab.append(float(fut.get(e, 0) >= 3))
    summ = {k: float(np.nanmean([r[k] for r in recs])) for k in recs[0]}
    summ["per_page"] = {k: [float(r[k]) for r in recs] for k in ("gain", "recall", "reward", "diversity", "forwards")}
    if time_it:
        summ["per_page"]["latency_ms"] = [float(x) for x in lat]
    summ["violations_total"] = int(sum(r["violations"] for r in recs))
    if time_it:
        summ["latency_ms_mean"] = float(np.mean(lat))
        summ["latency_ms_p50"] = float(np.median(lat))
        summ["latency_ms_p95"] = float(np.percentile(lat, 95))
    drafted = sum(sum(o.drafted.values()) for o in outs)
    if drafted:
        summ["acceptance"] = sum(sum(o.accepted.values()) for o in outs) / drafted
        by_pos = {}
        for o in outs:
            for p, n in o.drafted.items():
                d = by_pos.setdefault(p, [0, 0])
                d[0] += n
                d[1] += o.accepted.get(p, 0)
        summ["acceptance_by_draft_pos"] = {int(p): a / n for p, (n, a) in sorted(by_pos.items())}
    if hasattr(outs[0], "conflicts"):
        summ["conflicts_per_page"] = float(np.mean([o.conflicts for o in outs]))
    summ["mean_k_used"] = float(np.mean([np.mean(o.k_used) for o in outs]))
    summ["value_ece"] = ece(conf, lab)
    summ["arm"] = name
    print(f"  {name:<32} gain {summ['gain']:.4f} recall {summ['recall']:.4f} div {summ['diversity']:.3f} "
          f"fwd {summ['forwards']:.1f} lat {summ.get('latency_ms_mean', 0):.1f}ms viol {summ['violations_total']}"
          + (f" acc {summ['acceptance']:.3f}" if "acceptance" in summ else "")
          + (f" slot_agree {summ['slot_agree']:.4f} exact {summ['page_exact']:.3f}" if baseline else ""))
    return summ, outs


def teacher_pages(imps):
    """Logged production (teacher) pages wrapped as PageOut-like objects for reference metrics."""
    from genpage.decoding.page import PageOut
    return [PageOut(rows=imp["rows"].tolist(), ents=imp["ents"].tolist(),
                    values=[[0.0] * L for _ in range(R)], k_used=[L] * R) for imp in imps]


def save(name, obj):
    if TINY:
        name += "_tiny"
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote results/{name}.json")


def devices(arg):
    return [torch.device(d) for d in arg.split(",")]
