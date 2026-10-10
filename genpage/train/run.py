"""Train GenPage stages.

  python -m genpage.train.run --stage pretrain --lam 1 --size S
  python -m genpage.train.run --stage wbc      --lam 1 --size S   # initialised from pretrain ckpt
  python -m genpage.train.run --stage wbc_scratch ...              # Table-1 ablation: no pretraining
"""
import argparse
import json

from genpage.config import RESULTS_DIR, DataConfig, TrainConfig
from genpage.train.core import (Examples, build_model, ckpt_path, evaluate, load_ckpt, load_split, ntp_loss,
                                save_ckpt, train_loop, wbc_loss)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "wbc", "wbc_scratch"], required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--hist-cap", type=int, default=100)
    ap.add_argument("--size", default="S")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for checkpoint names (e.g. a training-budget control)")
    ap.add_argument("--frac", type=float, default=1.0, help="train on this fraction of training users")
    a = ap.parse_args(argv)

    dcfg = DataConfig(mmr_lambda=a.lam, hist_cap=a.hist_cap)
    tok, split = load_split(dcfg, a.tiny)
    pre = a.stage == "pretrain"
    tcfg = TrainConfig(epochs=a.epochs or (4 if pre else 2), lr=a.lr or (1e-3 if pre else 3e-4))
    if a.tiny:
        tcfg.max_steps, tcfg.log_every, tcfg.warmup_steps = 60, 20, 10

    if a.frac < 1.0:  # data-scarce (cold-start) setting: keep a random subset of training users
        import numpy as np
        users = sorted({i["user"] for i in split["train"]})
        keep = set(np.random.default_rng(0).choice(users, size=max(1, int(len(users) * a.frac)), replace=False).tolist())
        split["train"] = [i for i in split["train"] if i["user"] in keep]
    train = Examples(split["train"], tok, positive_only=pre)
    val = Examples(split["val"], tok)
    test = Examples(split["test"], tok)
    if a.stage == "wbc":
        model = load_ckpt(ckpt_path(dcfg, a.size, "pretrain" + a.tag, a.tiny)).train()
    else:
        model = build_model(a.size, tok.vocab_size, max_len=1024 if a.hist_cap <= 300 else 2048)
    print(f"{a.stage}: {len(train)} train examples, model {a.size} "
          f"{model.n_params() / 1e6:.2f}M non-emb / {model.n_params(False) / 1e6:.2f}M total")

    hist = train_loop(model, train, ntp_loss if pre else wbc_loss, tok, tcfg, val)
    out = ckpt_path(dcfg, a.size, a.stage + a.tag, a.tiny)
    save_ckpt(model, out, stage=a.stage, data=dcfg.__dict__)
    m = evaluate(model, test, tok, tcfg)
    print(f"test: {m}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"train_{out.stem}.json").write_text(json.dumps(dict(test=m, val_history=hist), indent=1))


if __name__ == "__main__":
    main()
