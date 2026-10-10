def test_page_roundtrip(data):
    tok, split = data
    for imp in split["test"][:200]:
        rows, ents = tok.decode_page(tok.encode_page(imp["rows"], imp["ents"]))
        assert rows == imp["rows"].tolist()
        assert ents == imp["ents"].tolist()


def test_context_layout(data):
    tok, split = data
    ctx = split["train"][0]["ctx"].tolist()
    assert ctx[0] == tok.BOS and ctx[-1] == tok.PAGE
    hist = ctx[ctx.index(tok.HIST) + 1:-1]
    assert len(hist) % 3 == 0
    for e, r, d in zip(hist[0::3], hist[1::3], hist[2::3]):
        assert tok.is_ent(e) and tok.rating0 <= r < tok.rating0 + 5 and tok.dt0 <= d < tok.gender0
