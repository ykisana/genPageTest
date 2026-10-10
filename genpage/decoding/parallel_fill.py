"""Decoders for the output-changing parallel arms.

decode_skeleton (E5): R skeleton steps, then all rows filled as a batch (k AR steps + 1 tail pass).
decode_parallel (E7): one forward over [ctx][slot queries]; constraints applied by greedy assignment.

Cross-row duplicates cannot be prevented inside a parallel step, so both resolve them top-down
(row 0 wins, later rows take their next-best eligible entity) and report how often that happened.
"""
import torch

from genpage.decoding.constraints import Constraints, masked
from genpage.decoding.page import PageOut, Session
from genpage.eval.profile import NULL


def _resolve(cands, taken, n_needed):
    """cands: list of ranked candidate lists. Picks n_needed per list top-down, skipping `taken`.
    Returns (picks, conflicts) where a conflict = a pick that was not the list's best remaining choice."""
    picks, conflicts = [], 0
    for lst in cands:
        got = []
        for e in lst:
            if e in taken:
                continue
            got.append(e)
            taken.add(e)
            if len(got) == n_needed:
                break
        # a conflict is any slot whose unconstrained top choice was claimed by an earlier row
        conflicts += sum(1 for e in lst[:n_needed] if e not in got)
        if len(got) < n_needed:
            raise RuntimeError("candidate list exhausted; increase the candidate pool")
        picks.append(got)
    return picks, conflicts


@torch.no_grad()
def decode_skeleton(model, tok, ctx, cons: Constraints, R, L, k, prof=NULL):
    sess = Session(model, len(ctx) + R + L + 4, prof)
    rs, es = tok.row_slice(), tok.ent_slice()
    out = PageOut()
    out.conflicts = 0
    logits, _ = sess.fwd(ctx, "prefill")
    cur = logits[-1]
    rows = []
    for _ in range(R):
        with prof.section("mask"):
            s = masked(cur[rs], cons.row_mask())
        row = int(s.argmax())
        cons.add_row(row)
        rows.append(row)
        logits, _ = sess.fwd([tok.row_tok(row)])
        cur = logits[-1]

    P0 = len(ctx) + R
    dev = sess.dev
    with prof.section("expand"):
        sess.cache.expand(R)
    row_idx = torch.arange(R, device=dev).view(R, 1)
    elig = cons.ct.elig[torch.as_tensor(rows, device=dev)]           # [R, NE]

    def step(tokens, p):
        with prof.section("forward"):
            x = torch.as_tensor(tokens, dtype=torch.long, device=dev).view(R, 1)
            pos = torch.full((R, 1), p, dtype=torch.long, device=dev)
            h = model.hidden(x, sess.cache, pos=pos, row_idx=row_idx)
            sess.forwards += 1
            return model.head(h[:, -1])

    cur = step([tok.row_tok(r) for r in rows], P0)
    ents = [[] for _ in range(R)]
    vals = [[] for _ in range(R)]
    k = min(k, L)
    for j in range(k):
        with prof.section("mask"):
            s = masked(cur[:, es], elig & cons.avail)
        with prof.section("select"):
            top = s.topk(R + 1, dim=-1)
            cands = top.indices.tolist()
        picks, c = _resolve(cands, set(), 1)
        out.conflicts += c
        for i in range(R):
            ents[i].append(picks[i][0])
            vals[i].append(float(cur[i, tok.ent0 + picks[i][0]]))
        cons.add_ents([p[0] for p in picks])
        if j < L - 1:
            cur = step([tok.ent_tok(p[0]) for p in picks], P0 + 1 + j)
    n_tail = L - k
    if n_tail > 0:
        with prof.section("mask"):
            s = masked(cur[:, es], elig & cons.avail)
        with prof.section("select"):
            top = s.topk(min(n_tail * (R + 1), s.shape[1]), dim=-1)
            cands = top.indices.tolist()
        picks, c = _resolve(cands, set(), n_tail)
        out.conflicts += c
        for i in range(R):
            ents[i].extend(picks[i])
            vals[i].extend(float(cur[i, tok.ent0 + e]) for e in picks[i])
        cons.add_ents([e for p in picks for e in p])
    out.rows, out.ents, out.values = rows, ents, vals
    out.k_used = [k] * R
    out.forwards = sess.forwards
    return out


@torch.no_grad()
def decode_parallel(model, tok, ctx, cons: Constraints, R, L, joint: bool, prof=NULL):
    from genpage.train.layouts import ParallelExamples
    dev = next(model.parameters()).device
    n_slots = R * (1 + L)
    C = len(ctx)
    T = C + n_slots
    out = PageOut()
    out.conflicts = 0
    with prof.section("prefill"):
        x = torch.tensor([list(ctx) + [tok.PAD] * n_slots], dtype=torch.long, device=dev)
        ridx = torch.full((1, T), -1, dtype=torch.long, device=dev)
        ridx[0, C:] = torch.arange(n_slots, device=dev)
        mask = torch.from_numpy(ParallelExamples.mask_for(C, n_slots, joint, T)).to(dev).view(1, 1, T, T)
        h = model.hidden(x, pos=torch.arange(T, device=dev), attn_mask=mask, row_idx=ridx)[0, C:]
        logits = model.head(h)                                       # [n_slots, V]
    out.forwards = 1
    slot_of = lambda r, c: r * (1 + L) + 1 + c
    rs, es = tok.row_slice(), tok.ent_slice()
    taken = set()
    for r in range(R):
        with prof.section("mask"):
            s = masked(logits[r * (1 + L), rs], cons.row_mask())
            row = int(s.argmax())
        cons.add_row(row)
        with prof.section("mask"):
            s = masked(logits[[slot_of(r, c) for c in range(L)]][:, es], cons.ent_mask(row).unsqueeze(0))
        with prof.section("select"):
            cands = s.topk(L + 1, dim=-1).indices.tolist()
        picks, c = _resolve(cands, taken, 1)
        out.conflicts += c
        row_ents = [p[0] for p in picks]
        cons.add_ents(row_ents)
        out.rows.append(row)
        out.ents.append(row_ents)
        out.values.append([float(logits[slot_of(r, c), tok.ent0 + e]) for c, e in enumerate(row_ents)])
    out.k_used = [0] * R
    return out
