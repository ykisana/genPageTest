"""Batch-1 page decoders built on one shared loop, so every strategy uses identical row
selection, constraint masks and tail ranking. Only how the first k entities of a row are
produced differs:

  fill="ar"          GenPage hybrid row decoding: k entities one forward at a time, then the tail
  fill="spec"        self-speculative: draft = current masked ranking, verify in one forward (lossless)
  fill="mtp_spec"    draft from MTP heads, verify in one forward (lossless)
  fill="mtp_direct"  accept MTP drafts without verification (lossy)
  fill="gated"       stop autoregressing early when a calibrated gate says the ranking is stable (lossy)
  fill="oracle"      same pages as "ar"; counts forwards as if a perfect gate had stopped at the earliest
                     step whose tail reproduces the row exactly (upper bound for "gated")
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from genpage.decoding.constraints import Constraints, masked
from genpage.eval.profile import NULL


@dataclass
class PageOut:
    rows: list = field(default_factory=list)
    ents: list = field(default_factory=list)
    values: list = field(default_factory=list)      # raw logit of each chosen entity (WBC value)
    forwards: int = 0
    k_used: list = field(default_factory=list)      # AR entities per row before the tail
    drafted: dict = field(default_factory=dict)     # draft position -> count (position 0 is exact, not counted)
    accepted: dict = field(default_factory=dict)
    gate_records: list = field(default_factory=list)

    def acceptance(self):
        d, a = sum(self.drafted.values()), sum(self.accepted.values())
        return a / d if d else float("nan")


class Session:
    """KV cache + forward bookkeeping for one request."""

    def __init__(self, model, max_len, prof):
        self.model, self.prof = model, prof
        self.dev = next(model.parameters()).device
        self.cache = model.new_cache(1, max_len)
        self.forwards = 0

    @torch.no_grad()
    def fwd(self, toks, phase="forward"):
        with self.prof.section(phase):
            x = torch.as_tensor(toks, dtype=torch.long, device=self.dev).view(1, -1)
            h = self.model.hidden(x, self.cache)[0]
            logits = self.model.head(h)
        self.forwards += 1
        return logits, h


def gate_features(ent_scores: torch.Tensor, j: int) -> torch.Tensor:
    """Features of the current masked entity ranking used by the confidence gate (E6)."""
    top = ent_scores.topk(3).values
    s1, s2, s3 = top[0], top[1], top[2]
    return torch.stack([s1, s2, s3, s1 - s2, s2 - s3, torch.sigmoid(s1), torch.sigmoid(s2),
                        torch.tensor(float(j), device=top.device)])


@torch.no_grad()
def decode_page(model, tok, ctx, cons: Constraints, R, L, k, fill="ar", prof=NULL,
                gate=None, gate_threshold=0.5, record_gate=False):
    sess = Session(model, len(ctx) + R * (L + 1) + 8, prof)
    rs, es = tok.row_slice(), tok.ent_slice()
    out = PageOut()
    logits, hid = sess.fwd(ctx, "prefill")
    cur, cur_h = logits[-1], hid[-1]
    k = min(k, L)

    def pick(scores, mask):
        with prof.section("mask"):
            s = masked(scores, mask)
        with prof.section("select"):
            return int(s.argmax())

    for r in range(R):
        row = pick(cur[rs], cons.row_mask())
        cons.add_row(row)
        logits, hid = sess.fwd([tok.row_tok(row)])
        cur, cur_h = logits[-1], hid[-1]
        placed, vals = [], []

        if fill in ("ar", "gated", "oracle"):
            # states j>=1 where the decoder could stop and take the tail: (j, logits, features)
            track = record_gate or fill == "oracle"
            states = []
            avail0 = cons.avail.clone() if track else None
            for j in range(k):
                with prof.section("mask"):
                    s = masked(cur[es], cons.ent_mask(row))
                with prof.section("select"):
                    e = int(s.argmax())
                if j >= 1 and (track or fill == "gated"):
                    feats = gate_features(s, j)
                    if track:
                        states.append((j, cur, feats))
                    if fill == "gated" and gate(feats) >= gate_threshold:
                        break
                cons.add_ents([e])
                placed.append(e)
                vals.append(float(cur[tok.ent0 + e]))
                logits, hid = sess.fwd([tok.ent_tok(e)])
                cur, cur_h = logits[-1], hid[-1]

        else:  # speculative / MTP strategies
            need = k
            while need > 0:
                with prof.section("mask"):
                    base = cons.ent_mask(row)
                    s = masked(cur[es], base)
                with prof.section("draft"):
                    if fill == "spec":
                        draft = s.topk(need).indices.tolist()
                    else:
                        draft = _mtp_draft(model, cur_h, s, base, es, need)
                logits, hid = sess.fwd([tok.ent_tok(e) for e in draft])
                m = len(draft)
                if fill == "mtp_direct":
                    n_acc = m
                else:
                    with prof.section("verify"):
                        n_acc = _verify(logits, draft, base, es)
                for p in range(1, m):
                    out.drafted[p] = out.drafted.get(p, 0) + 1
                    out.accepted[p] = out.accepted.get(p, 0) + int(p < n_acc)
                vals.append(float(cur[tok.ent0 + draft[0]]))
                vals.extend(float(logits[i - 1, tok.ent0 + draft[i]]) for i in range(1, n_acc))
                sess.cache.truncate(sess.cache.len - (m - n_acc))
                cons.add_ents(draft[:n_acc])
                placed.extend(draft[:n_acc])
                cur, cur_h = logits[n_acc - 1], hid[n_acc - 1]
                need -= n_acc

        out.k_used.append(len(placed))
        n_tail = L - len(placed)
        if n_tail > 0:
            with prof.section("mask"):
                s = masked(cur[es], cons.ent_mask(row))
            with prof.section("select"):
                top = s.topk(n_tail)
                tail = top.indices.tolist()
                vals.extend(top.values.tolist())
            cons.add_ents(tail)
            placed.extend(tail)
            if r < R - 1:  # the last row's tail does not need to be fed
                logits, hid = sess.fwd([tok.ent_tok(e) for e in tail])
                cur, cur_h = logits[-1], hid[-1]
        if fill in ("ar", "gated", "oracle") and track and states:
            exact = _stop_exact(states, placed, avail0, cons.ct.elig[row], es, L)
            if record_gate:
                out.gate_records.extend((f.cpu().numpy(), int(x), r, j) for (j, _, f), x in zip(states, exact))
            if fill == "oracle":
                # a perfect gate stops at the first state whose tail reproduces the full-k row exactly;
                # the page is identical, only the forwards for e_{j+1..k} are saved
                j_min = next((j for (j, _, _), x in zip(states, exact) if x), k)
                sess.forwards -= k - j_min
                out.k_used[-1] = j_min
        out.rows.append(row)
        out.ents.append(placed)
        out.values.append(vals)
    out.forwards = sess.forwards
    return out


def _verify(logits, draft, base_mask, es) -> int:
    """Longest prefix of `draft` that greedy decoding would also produce. draft[0] is exact.

    Row i of `logits` predicts the token after draft[i]; its mask excludes draft[:i+1].
    """
    m = len(draft)
    if m == 1:
        return 1
    s = logits[: m - 1, es]
    d = torch.as_tensor(draft, device=s.device)
    mask = base_mask.unsqueeze(0).expand(m - 1, -1).clone()
    excl = torch.tril(torch.ones(m - 1, m - 1, dtype=torch.bool, device=s.device))  # row i excludes draft[0..i]
    rows_i, cols_j = excl.nonzero(as_tuple=True)
    mask[rows_i, d[cols_j]] = False
    pred = masked(s, mask).argmax(-1).tolist()
    n = 1
    for i in range(m - 1):
        if pred[i] != draft[i + 1]:
            break
        n += 1
    return n


def _mtp_draft(model, h, main_scores, base_mask, es, need):
    draft = [int(main_scores.argmax())]
    excl = base_mask.clone()
    excl[draft[0]] = False
    for head in model.mtp[: need - 1]:
        s = masked(head(h)[es], excl)
        e = int(s.argmax())
        draft.append(e)
        excl[e] = False
    return draft


def _stop_exact(states, full_row, avail0, elig_row, es, L):
    """For each stop state j: would taking the tail from its logits reproduce the full-k row?"""
    res = []
    for j, logits_j, _ in states:
        mask = elig_row & avail0
        mask[torch.as_tensor(full_row[:j], device=mask.device)] = False
        tail = masked(logits_j[es], mask).topk(L - j).indices.tolist()
        res.append(tail == full_row[j:])
    return res
