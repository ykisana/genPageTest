"""Alternative sequence layouts for the output-changing arms.

Skeleton (E5):  [ctx][r1..rR][r1 e11..e1L][r2 e21..e2L]...   each block sees ctx+skeleton+itself only;
                blocks share position ids and carry a row-index embedding.
Parallel (E7):  [ctx][slot_0 .. slot_{R(1+L)-1}]               slots are learned queries (row_idx emb on <PAD>);
                variant "indep": slots see ctx + themselves; variant "joint": slots also see each other.
"""
import numpy as np
import torch

from genpage.train.core import Examples, page_targets


def row_rewards(imp):
    return imp["reward"].sum(1)


class SkeletonExamples(Examples):
    def __init__(self, imps, tok, R, L, positive_only=False):
        super().__init__(imps, tok, positive_only)
        self.R, self.L, self.tok = R, L, tok
        self.lay = []
        for i, imp in enumerate(self.imps):
            C = len(imp["ctx"])
            rows, ents, rew = imp["rows"], imp["ents"], imp["reward"]
            seq = list(imp["ctx"]) + [tok.row_tok(r) for r in rows]
            pos = list(range(C + R))
            ridx = [-1] * (C + R)
            block = [-1] * (C + R)
            trew = [np.nan] * (C + R)                 # reward of the token at this index (as a target)
            istgt = [False] * (C + R)
            for j in range(R):
                istgt[C + j] = True
                trew[C + j] = float(rew[j].sum())
            P0 = C + R
            for b in range(R):
                seq.append(tok.row_tok(rows[b])); pos.append(P0); ridx.append(b); block.append(b)
                trew.append(np.nan); istgt.append(False)
                for c in range(L):
                    seq.append(tok.ent_tok(ents[b][c])); pos.append(P0 + 1 + c); ridx.append(b); block.append(b)
                    trew.append(float(rew[b][c])); istgt.append(True)
            self.seqs[i] = np.array(seq, dtype=np.int32)
            self.lay.append((np.array(pos), np.array(ridx), np.array(block), np.array(trew, dtype=np.float32),
                             np.array(istgt), C + R))
        self.lens = np.array([len(s) for s in self.seqs])

    @staticmethod
    def mask_for(block: np.ndarray, prefix: int) -> np.ndarray:
        T = len(block)
        q = np.arange(T)[:, None]
        k = np.arange(T)[None, :]
        causal = k <= q
        same_block = (block[:, None] == block[None, :]) & (block[None, :] >= 0)
        key_prefix = k < prefix
        return causal & (key_prefix | same_block)

    def collate(self, ids):
        T = int(self.lens[ids].max())
        B = len(ids)
        toks = np.zeros((B, T), dtype=np.int64)
        pos = np.zeros((B, T), dtype=np.int64)
        ridx = np.full((B, T), -1, dtype=np.int64)
        rew = np.full((B, T - 1), np.nan, dtype=np.float32)
        tgt = np.zeros((B, T - 1), dtype=bool)
        mask = np.zeros((B, 1, T - 1, T - 1), dtype=bool)
        for b, i in enumerate(ids):
            s = self.seqs[i]
            p, ri, blk, trew, istgt, prefix = self.lay[i]
            n = len(s)
            toks[b, :n] = s
            pos[b, :n] = p
            pos[b, n:] = np.arange(p[-1] + 1, p[-1] + 1 + T - n)
            ridx[b, :n] = ri
            tgt[b, :n - 1] = istgt[1:]
            rew[b, :n - 1] = trew[1:]
            full_blk = np.concatenate([blk, np.full(T - n, -2)])
            m = self.mask_for(full_blk, prefix)
            m[n:, :] = False
            m[np.arange(n, T), np.arange(n, T)] = True     # padding rows attend to themselves
            mask[b, 0] = m[: T - 1, : T - 1]
        return dict(x=torch.from_numpy(toks[:, :-1]), y=torch.from_numpy(toks[:, 1:]),
                    tgt=torch.from_numpy(tgt), rew=torch.from_numpy(rew),
                    pos=torch.from_numpy(pos[:, :-1]), row_idx=torch.from_numpy(ridx[:, :-1]),
                    attn_mask=torch.from_numpy(mask), ids=ids)


class ParallelExamples(Examples):
    """Slot queries appended after the context. Targets live at the slot positions themselves,
    so x == full sequence and y holds the teacher token for each slot."""

    def __init__(self, imps, tok, R, L, joint: bool, positive_only=False):
        super().__init__(imps, tok, positive_only)
        self.R, self.L, self.tok, self.joint = R, L, tok, joint
        self.n_slots = R * (1 + L)
        self.ctx_only = [np.asarray(imp["ctx"], dtype=np.int32) for imp in self.imps]
        self.lens = np.array([len(c) + self.n_slots for c in self.ctx_only])
        self.slot_tok, self.slot_rew = [], []
        for imp in self.imps:
            ptoks, prew = page_targets(imp, tok)
            self.slot_tok.append(np.array(ptoks[:-1], dtype=np.int64))   # drop EOS
            self.slot_rew.append(np.array(prew[:-1], dtype=np.float32))

    @staticmethod
    def mask_for(C: int, n_slots: int, joint: bool, T: int) -> np.ndarray:
        q = np.arange(T)[:, None]
        k = np.arange(T)[None, :]
        m = (k <= q) & (k < C)                      # causal over the context
        is_slot_q = (q >= C) & (q < C + n_slots)
        m |= is_slot_q & (k < C)                    # slots see the whole context
        if joint:
            m |= is_slot_q & (k >= C) & (k < C + n_slots)
        else:
            m |= is_slot_q & (k == q)
        m[np.arange(T), np.arange(T)] = True
        return m

    def collate(self, ids):
        T = int(self.lens[ids].max())
        B = len(ids)
        toks = np.zeros((B, T), dtype=np.int64)
        ridx = np.full((B, T), -1, dtype=np.int64)
        y = np.zeros((B, T), dtype=np.int64)
        rew = np.full((B, T), np.nan, dtype=np.float32)
        tgt = np.zeros((B, T), dtype=bool)
        mask = np.zeros((B, 1, T, T), dtype=bool)
        for b, i in enumerate(ids):
            c = self.ctx_only[i]
            C = len(c)
            toks[b, :C] = c
            toks[b, C:C + self.n_slots] = self.tok.PAD
            ridx[b, C:C + self.n_slots] = np.arange(self.n_slots)
            y[b, C:C + self.n_slots] = self.slot_tok[i]
            rew[b, C:C + self.n_slots] = self.slot_rew[i]
            tgt[b, C:C + self.n_slots] = True
            mask[b, 0] = self.mask_for(C, self.n_slots, self.joint, T)
        return dict(x=torch.from_numpy(toks), y=torch.from_numpy(y), tgt=torch.from_numpy(tgt),
                    rew=torch.from_numpy(rew), row_idx=torch.from_numpy(ridx),
                    pos=torch.arange(T).unsqueeze(0).expand(B, T).contiguous(),
                    attn_mask=torch.from_numpy(mask), ids=ids)
