import torch

from genpage.decoding.constraints import CatalogTensors, Constraints
from genpage.decoding.page import decode_page
from genpage.eval.metrics import violations

R, L = 6, 10


def _run(model, tok, cat, imp, **kw):
    ct = CatalogTensors(cat, torch.device("cpu"))
    return decode_page(model, tok, imp["ctx"].tolist(), Constraints(ct, imp["watched"], L), R, L, **kw)


def test_no_violations(rand_model, data, cat):
    tok, split = data
    for imp in split["test"][:10]:
        for fill in ["ar", "spec", "mtp_spec", "mtp_direct"]:
            out = _run(rand_model, tok, cat, imp, k=3, fill=fill)
            assert violations(out.rows, out.ents, imp, cat, L) == 0, fill


def test_spec_is_lossless(rand_model, data, cat):
    tok, split = data
    for imp in split["test"][:20]:
        for k in (3, 10):
            base = _run(rand_model, tok, cat, imp, k=k, fill="ar")
            for fill in ("spec", "mtp_spec"):
                out = _run(rand_model, tok, cat, imp, k=k, fill=fill)
                assert out.rows == base.rows and out.ents == base.ents, (fill, k)
                assert out.forwards <= base.forwards


def test_forward_counts(rand_model, data, cat):
    tok, split = data
    imp = split["test"][0]
    base = _run(rand_model, tok, cat, imp, k=3, fill="ar")
    # prefill + per row: row + k entities + tail (no tail feed on the last row)
    assert base.forwards == 1 + R * (1 + 3 + 1) - 1


def test_gate_extremes(rand_model, data, cat):
    tok, split = data
    for imp in split["test"][:5]:
        never = _run(rand_model, tok, cat, imp, k=3, fill="gated", gate=lambda f: 0.0, gate_threshold=float("inf"))
        k3 = _run(rand_model, tok, cat, imp, k=3, fill="ar")
        assert never.ents == k3.ents and never.forwards == k3.forwards
        always = _run(rand_model, tok, cat, imp, k=3, fill="gated", gate=lambda f: 1.0, gate_threshold=0.0)
        k1 = _run(rand_model, tok, cat, imp, k=1, fill="ar")
        assert always.ents == k1.ents and always.forwards == k1.forwards


def test_oracle_is_exact_and_cheaper(rand_model, data, cat):
    tok, split = data
    for imp in split["test"][:5]:
        base = _run(rand_model, tok, cat, imp, k=5, fill="ar")
        orc = _run(rand_model, tok, cat, imp, k=5, fill="oracle", record_gate=True)
        assert orc.ents == base.ents and orc.forwards <= base.forwards
        assert len(orc.gate_records) == R * 4
