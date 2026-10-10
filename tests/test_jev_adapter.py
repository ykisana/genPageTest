import torch

from genpage.decoding.constraints import CatalogTensors, Constraints
from genpage.eval.metrics import violations
from genpage.jev_adapter import JevDecision

R, L = 6, 10


def test_adapter_with_mock_transport(rand_model, data, cat):
    tok, split = data
    calls = []

    def mock(payload):
        calls.append(payload)
        # probabilities favour the *last* options and include a bogus id, to exercise the fix-up path
        return {"answers": {q["id"]: {"-1": 0.9, **{o["id"]: (i + 1) / 1000 for i, o in enumerate(q["options"])}}
                            for q in payload["questions"]}}

    jev = JevDecision(mock, rand_model, tok, cat, R, L)
    ct = CatalogTensors(cat, torch.device("cpu"))
    imp = split["test"][0]
    out = jev.decide_page(imp, Constraints(ct, imp["watched"], L))
    assert len(calls) == 1 and len(calls[0]["questions"]) == R
    assert violations(out.rows, out.ents, imp, cat, L) == 0
    assert out.invalid == R            # every row had one bogus answer


def _mock_prefer_top(payload):
    # rank options in the order given (GenPage/popularity order)
    return {"answers": {q["id"]: {o["id"]: 1.0 / (i + 1) for i, o in enumerate(q["options"])}
                        for q in payload["questions"]}}


def test_dedup_has_no_collisions(rand_model, data, cat):
    tok, split = data
    ct = CatalogTensors(cat, torch.device("cpu"))
    imp = split["test"][1]
    jev = JevDecision(_mock_prefer_top, rand_model, tok, cat, R, L, dedup=True)
    out = jev.decide_page(imp, Constraints(ct, imp["watched"], L))
    assert out.conflicts == 0 and violations(out.rows, out.ents, imp, cat, L) == 0


def test_sequential_one_call_per_row_and_sees_page(rand_model, data, cat):
    tok, split = data
    calls = []
    jev = JevDecision(lambda p: (calls.append(p), _mock_prefer_top(p))[1], rand_model, tok, cat, R, L,
                      sequential=True)
    ct = CatalogTensors(cat, torch.device("cpu"))
    imp = split["test"][2]
    out = jev.decide_page(imp, Constraints(ct, imp["watched"], L))
    assert len(calls) == R and "already_on_page_above" in calls[1]["context"]
    assert len(calls[-1]["context"]["already_on_page_above"]) == (R - 1) * L
    assert out.conflicts == 0 and violations(out.rows, out.ents, imp, cat, L) == 0


def test_popular_source_needs_no_model(data, cat):
    tok, split = data
    ct = CatalogTensors(cat, torch.device("cpu"))
    imp = split["test"][3]
    jev = JevDecision(_mock_prefer_top, None, tok, cat, R, L, source="popular", dedup=True)
    out = jev.decide_page(imp, Constraints(ct, imp["watched"], L))
    assert out.forwards == 0 and violations(out.rows, out.ents, imp, cat, L) == 0
