"""Shared training utilities: example construction, length-bucketed batching, loop, metrics."""
import math
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from genpage.config import CKPT_DIR, DataConfig, ModelConfig, TrainConfig
from genpage.data.build import load as load_data
from genpage.model import GenPageLM
from genpage.tokenizer import Tokenizer


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def page_targets(imp, tok: Tokenizer):
    """Page tokens plus a per-token reward (row reward = sum of its entities; EOS = nan)."""
    toks = tok.encode_page(imp["rows"], imp["ents"])
    rew = []
    for r in range(len(imp["rows"])):
        rew.append(float(imp["reward"][r].sum()))
        rew.extend(imp["reward"][r].tolist())
    rew.append(float("nan"))
    return toks, rew


class Examples:
    """Tokenized impressions for one split, with length-bucketed batching."""

    def __init__(self, imps, tok: Tokenizer, positive_only=False):
        if positive_only:  # pretraining: pages that received positive engagement
            imps = [i for i in imps if (i["reward"] > 0).any()]
        self.imps = imps
        self.seqs, self.ctx_len, self.rew = [], [], []
        for imp in imps:
            ptoks, prew = page_targets(imp, tok)
            self.seqs.append(np.concatenate([imp["ctx"], np.array(ptoks, dtype=np.int32)]))
            self.ctx_len.append(len(imp["ctx"]))
            self.rew.append(np.array(prew, dtype=np.float32))
        self.lens = np.array([len(s) for s in self.seqs])

    def __len__(self):
        return len(self.seqs)

    def batches(self, bs, shuffle, rng=None):
        idx = np.arange(len(self))
        if shuffle:
            idx = rng.permutation(idx)
            chunks = [idx[i:i + bs * 50] for i in range(0, len(idx), bs * 50)]
            idx = np.concatenate([c[np.argsort(self.lens[c], kind="stable")] for c in chunks])
            groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
            for g in rng.permutation(len(groups)):
                yield self.collate(groups[g])
        else:
            idx = idx[np.argsort(self.lens, kind="stable")]
            for i in range(0, len(idx), bs):
                yield self.collate(idx[i:i + bs])

    def collate(self, ids):
        T = int(self.lens[ids].max())
        toks = np.zeros((len(ids), T), dtype=np.int64)
        rew = np.full((len(ids), T - 1), np.nan, dtype=np.float32)
        tgt = np.zeros((len(ids), T - 1), dtype=bool)
        for b, i in enumerate(ids):
            s, c = self.seqs[i], self.ctx_len[i]
            toks[b, :len(s)] = s
            tgt[b, c - 1:len(s) - 1] = True         # predicting page tokens (incl. EOS)
            rew[b, c - 1:len(s) - 1] = self.rew[i]
        t = torch.from_numpy(toks)
        return dict(x=t[:, :-1], y=t[:, 1:], tgt=torch.from_numpy(tgt), rew=torch.from_numpy(rew), ids=ids)


def to_dev(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def target_logits(model, batch):
    """Logits only at page-target positions: returns (logits [N,V], y [N], rew [N])."""
    extra = {k: batch[k] for k in ("pos", "attn_mask", "row_idx") if k in batch}
    h = model.hidden(batch["x"], **extra)
    m = batch["tgt"]
    return model.head(h[m]), batch["y"][m], batch["rew"][m]


def ntp_loss(model, batch, tok, tcfg):
    logits, y, _ = target_logits(model, batch)
    return F.cross_entropy(logits, y)


def wbc_terms(logits, y, rew, tok: Tokenizer, n_neg: int, gen=None):
    """Weighted BCE on target-token logits + random same-type negatives."""
    keep = ~torch.isnan(rew)                      # drop EOS
    logits, y, rew = logits[keep], y[keep], rew[keep]
    zpos = logits.gather(1, y[:, None]).squeeze(1)
    label = (rew > 0).float()
    w = rew.abs().clamp(0.1, 2.0)
    is_row = (y >= tok.row0) & (y < tok.row0 + tok.n_row_types)
    lo = torch.where(is_row, tok.row0, tok.ent0)
    n = torch.where(is_row, tok.n_row_types, tok.n_entities)
    u = torch.rand((len(y), n_neg), device=y.device, generator=gen)
    neg = lo[:, None] + (u * n[:, None]).long()
    zneg = logits.gather(1, neg)
    return zpos, label, w, zneg, is_row


def wbc_loss(model, batch, tok, tcfg):
    logits, y, rew = target_logits(model, batch)
    zpos, label, w, zneg, _ = wbc_terms(logits, y, rew, tok, tcfg.n_random_neg)
    lpos = F.binary_cross_entropy_with_logits(zpos, label, weight=w, reduction="sum")
    lneg = F.binary_cross_entropy_with_logits(zneg, torch.zeros_like(zneg), reduction="sum") * tcfg.neg_weight
    return (lpos + lneg) / (w.sum() + tcfg.neg_weight * zneg.numel())


def weighted_auc(scores, labels, weights) -> float:
    o = np.argsort(scores, kind="stable")
    s, l, w = scores[o], labels[o], weights[o]
    wn = np.where(l == 0, w, 0.0)
    wp = np.where(l == 1, w, 0.0)
    # handle ties by grouping equal scores
    uniq, start = np.unique(s, return_index=True)
    neg_grp = np.add.reduceat(wn, start)
    pos_grp = np.add.reduceat(wp, start)
    neg_below = np.cumsum(neg_grp) - neg_grp
    num = (pos_grp * (neg_below + 0.5 * neg_grp)).sum()
    den = wp.sum() * wn.sum()
    return float(num / den) if den > 0 else float("nan")


@torch.no_grad()
def evaluate(model, ex: Examples, tok, tcfg, bs=64, max_batches=0):
    """NTP loss, WBC loss, and sample-weighted row/entity AUC over logged page targets."""
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(1234)
    tot = dict(ntp=0.0, n=0, wbc_num=0.0, wbc_den=0.0)
    sc, lab, wt, isrow = [], [], [], []
    for bi, b in enumerate(ex.batches(bs, shuffle=False)):
        if max_batches and bi >= max_batches:
            break
        b = to_dev(b, dev)
        logits, y, rew = target_logits(model, b)
        tot["ntp"] += F.cross_entropy(logits, y, reduction="sum").item()
        tot["n"] += len(y)
        zpos, label, w, zneg, is_row = wbc_terms(logits, y, rew, tok, tcfg.n_random_neg, gen)
        tot["wbc_num"] += (F.binary_cross_entropy_with_logits(zpos, label, weight=w, reduction="sum")
                           + tcfg.neg_weight * F.binary_cross_entropy_with_logits(
                               zneg, torch.zeros_like(zneg), reduction="sum")).item()
        tot["wbc_den"] += (w.sum() + tcfg.neg_weight * zneg.numel()).item()
        sc.append(zpos.cpu()); lab.append(label.cpu()); wt.append(w.cpu()); isrow.append(is_row.cpu())
    sc, lab, wt, isrow = (torch.cat(a).numpy() for a in (sc, lab, wt, isrow))
    model.train()
    return dict(
        ntp_loss=tot["ntp"] / tot["n"],
        wbc_loss=tot["wbc_num"] / tot["wbc_den"],
        row_auc=weighted_auc(sc[isrow], lab[isrow], wt[isrow]),
        entity_auc=weighted_auc(sc[~isrow], lab[~isrow], wt[~isrow]),
    )


def train_loop(model, ex: Examples, loss_fn, tok, tcfg: TrainConfig, val: Examples = None, log=print):
    dev = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tcfg.lr, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(ex) / tcfg.batch_size)
    total = steps_per_epoch * tcfg.epochs
    if tcfg.max_steps:
        total = min(total, tcfg.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tcfg.warmup_steps)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    rng = np.random.default_rng(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    step, t0, run = 0, time.time(), 0.0
    history = []
    model.train()
    for ep in range(tcfg.epochs):
        for b in ex.batches(tcfg.batch_size, shuffle=True, rng=rng):
            b = to_dev(b, dev)
            loss = loss_fn(model, b, tok, tcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % tcfg.log_every == 0:
                log(f"ep {ep} step {step}/{total} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time() - t0) / step * 1000:.0f}ms/step")
            if step >= total:
                break
        if val is not None:
            m = evaluate(model, val, tok, tcfg, max_batches=50)
            m["epoch"] = ep
            history.append(m)
            log(f"epoch {ep} val {m}")
        if step >= total:
            break
    return history


def build_model(size: str, vocab: int, **kw) -> GenPageLM:
    return GenPageLM(ModelConfig.preset(size, vocab, **kw)).to(device())


def ckpt_path(dcfg: DataConfig, size: str, stage: str, tiny=False):
    return CKPT_DIR / f"{dcfg.tag()}_{size}_{stage}{'_tiny' if tiny else ''}.pt"


def save_ckpt(model, path, **meta):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cfg=asdict(model.cfg), state=model.state_dict(), **meta), path)


def load_ckpt(path, dev=None) -> GenPageLM:
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = GenPageLM(ModelConfig(**d["cfg"]))
    m.load_state_dict(d["state"])
    return m.to(dev or device()).eval()


def load_split(dcfg: DataConfig, tiny=False):
    d = load_data(dcfg, tiny)
    tok = Tokenizer(d["n_entities"])
    by = {"train": [], "val": [], "test": []}
    for imp in d["impressions"]:
        by[imp["split"]].append(imp)
    return tok, by
