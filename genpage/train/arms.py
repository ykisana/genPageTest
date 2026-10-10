"""Train the extra arms. All start from the GenPage S pretrain (or WBC) checkpoint so every arm
shares the same representation learning budget.

  python -m genpage.train.arms --arm mtp            --lam 1   # E4: Medusa heads on frozen WBC trunk
  python -m genpage.train.arms --arm skeleton       --lam 1   # E5
  python -m genpage.train.arms --arm parallel_indep --lam 1   # E7 (a)
  python -m genpage.train.arms --arm parallel_joint --lam 1   # E7 (b)
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from genpage.config import RESULTS_DIR, DataConfig, ModelConfig, TrainConfig
from genpage.model import GenPageLM
from genpage.train.core import (Examples, ckpt_path, device, evaluate, load_split, ntp_loss, save_ckpt,
                                to_dev, train_loop, wbc_loss)
from genpage.train.layouts import ParallelExamples, SkeletonExamples

R, L = 6, 10


def init_from(path, **cfg_over) -> GenPageLM:
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{**d["cfg"], **cfg_over})
    m = GenPageLM(cfg)
    missing, unexpected = m.load_state_dict(d["state"], strict=False)
    assert not unexpected, unexpected
    print(f"init from {path.name}; new params: {sorted({k.split('.')[0] for k in missing})}")
    return m.to(device())


def mtp_loss(model, batch, tok, tcfg):
    with torch.no_grad():
        h = model.hidden(batch["x"])
        main = model.head(h)[..., tok.ent_slice()]
    is_ent = (batch["y"] >= tok.ent0) & batch["tgt"]
    T = h.shape[1]
    total, n = 0.0, 0
    for i, head in enumerate(model.mtp):
        off = i + 1
        # head i at position t drafts the token main would emit at t+off (teacher-forced)
        valid = is_ent[:, :T - off] & is_ent[:, off:]
        if not valid.any():
            continue
        hs = h[:, :T - off][valid]
        target = F.softmax(main[:, off:][valid], dim=-1)
        pred = F.log_softmax(head(hs)[:, tok.ent_slice()], dim=-1)
        total = total + F.kl_div(pred, target, reduction="batchmean")
        n += 1
    return total / max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["mtp", "skeleton", "parallel_indep", "parallel_joint"], required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--size", default="S")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    dcfg = DataConfig(mmr_lambda=a.lam)
    tok, split = load_split(dcfg, a.tiny)
    tcfg = TrainConfig()
    if a.tiny:
        tcfg.max_steps, tcfg.log_every, tcfg.warmup_steps = 30, 10, 5
    log = {}

    if a.arm == "mtp":
        model = init_from(ckpt_path(dcfg, a.size, "wbc", a.tiny), n_mtp_heads=2)
        for n_, p in model.named_parameters():
            p.requires_grad = n_.startswith("mtp.")
        tcfg.epochs, tcfg.lr = 1, 1e-3
        train = Examples(split["train"], tok)
        log["history"] = train_loop(model, train, mtp_loss, tok, tcfg)
    else:
        if a.arm == "skeleton":
            make = lambda imps, pos=False: SkeletonExamples(imps, tok, R, L, positive_only=pos)
            n_slots = R
        else:
            joint = a.arm == "parallel_joint"
            make = lambda imps, pos=False: ParallelExamples(imps, tok, R, L, joint, positive_only=pos)
            n_slots = R * (1 + L)
        model = init_from(ckpt_path(dcfg, a.size, "pretrain", a.tiny), n_row_slots=n_slots)
        val = make(split["val"])
        # stage 1: layout pretraining (NTP / slot CE) on positively engaged pages
        tcfg.epochs, tcfg.lr = 1, 5e-4
        log["pretrain"] = train_loop(model, make(split["train"], True), ntp_loss, tok, tcfg, val)
        # stage 2: WBC post-training
        tcfg.epochs, tcfg.lr = 2, 3e-4
        log["wbc"] = train_loop(model, make(split["train"]), wbc_loss, tok, tcfg, val)
        log["test"] = evaluate(model, make(split["test"]), tok, tcfg)
        print("test", log["test"])

    out = ckpt_path(dcfg, a.size, a.arm, a.tiny)
    save_ckpt(model, out, stage=a.arm, data=dcfg.__dict__)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"train_{out.stem}.json").write_text(json.dumps(log, indent=1, default=float))


if __name__ == "__main__":
    main()
