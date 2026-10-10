import numpy as np
import pytest
import torch

from genpage.config import ModelConfig
from genpage.decoding.constraints import CatalogTensors, Constraints
from genpage.decoding.parallel_fill import decode_parallel, decode_skeleton
from genpage.eval.metrics import violations
from genpage.model import GenPageLM
from genpage.train.layouts import ParallelExamples, SkeletonExamples

R, L = 6, 10


def _model(tok, n_slots):
    torch.manual_seed(1)
    m = GenPageLM(ModelConfig.preset("tiny", tok.vocab_size, max_len=1024, n_row_slots=n_slots)).eval()
    with torch.no_grad():
        m.head.weight.mul_(50)
        m.row_idx_emb.weight.normal_(0, 1)
    return m


def _logits(m, b):
    with torch.no_grad():
        return m(b["x"], pos=b["pos"], attn_mask=b["attn_mask"], row_idx=b["row_idx"])[0]


def test_skeleton_blocks_are_isolated(data):
    tok, split = data
    m = _model(tok, R)
    imp = dict(split["test"][0])
    ex = SkeletonExamples([imp], tok, R, L)
    base = _logits(m, ex.collate([0]))
    imp2 = dict(imp, ents=imp["ents"].copy())
    imp2["ents"][3] = (imp2["ents"][3] + 7) % tok.n_entities     # perturb row 3's entities only
    pert = _logits(m, SkeletonExamples([imp2], tok, R, L).collate([0]))
    blk = ex.lay[0][2][:-1]
    for b in range(R):
        same = torch.allclose(base[blk == b], pert[blk == b], atol=1e-5)
        assert same == (b != 3), b


def test_parallel_indep_vs_joint(data):
    tok, split = data
    imp = split["test"][0]
    C = len(imp["ctx"])
    for joint in (False, True):
        m = _model(tok, R * (1 + L))
        b = ParallelExamples([imp], tok, R, L, joint).collate([0])
        base = _logits(m, b)
        b["x"][0, C + 5] = tok.ent_tok(42)                         # perturb one slot's query
        pert = _logits(m, b)
        others = [C + i for i in range(R * (1 + L)) if i != 5]
        assert torch.allclose(base[others], pert[others], atol=1e-5) == (not joint)


def test_skeleton_decode_matches_training_layout(data, cat):
    tok, split = data
    m = _model(tok, R)
    ct = CatalogTensors(cat, torch.device("cpu"))
    for imp in split["test"][:5]:
        out = decode_skeleton(m, tok, imp["ctx"].tolist(), Constraints(ct, imp["watched"], L), R, L, k=3)
        assert violations(out.rows, out.ents, imp, cat, L) == 0
        fake = dict(imp, rows=np.array(out.rows), ents=np.array(out.ents), reward=np.zeros((R, L), np.float32))
        ex = SkeletonExamples([fake], tok, R, L)
        lg = _logits(m, ex.collate([0]))
        seq = ex.seqs[0]
        # for every entity slot, the full-layout logit of the chosen entity == the decode-time value
        for i in range(len(seq) - 1):
            y = int(seq[i + 1])
            if ex.lay[0][2][i + 1] >= 0 and tok.is_ent(y):
                b = ex.lay[0][2][i + 1]
                c = ex.lay[0][0][i + 1] - ex.lay[0][0][ex.lay[0][2] == b].min() - 1
                if c < 3 or c == 3:  # AR slots and the first tail slot come from fed positions
                    assert abs(lg[i, y].item() - out.values[b][c]) < 1e-3, (b, c)


def test_parallel_decode_valid(data, cat):
    tok, split = data
    ct = CatalogTensors(cat, torch.device("cpu"))
    for joint in (False, True):
        m = _model(tok, R * (1 + L))
        for imp in split["test"][:5]:
            out = decode_parallel(m, tok, imp["ctx"].tolist(), Constraints(ct, imp["watched"], L), R, L, joint)
            assert out.forwards == 1
            assert violations(out.rows, out.ents, imp, cat, L) == 0
