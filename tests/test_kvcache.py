import torch


def test_cached_matches_uncached(rand_model, data):
    tok, split = data
    seq = torch.as_tensor(split["test"][0]["ctx"].tolist() + [tok.row_tok(0)] + [tok.ent_tok(i) for i in range(12)])
    m = rand_model
    with torch.no_grad():
        full = m(seq.view(1, -1))[0]
        cache = m.new_cache(1, 1024)
        n = len(seq) - 13
        parts = [m(seq[:n].view(1, -1), cache)[0]]
        parts.append(m(seq[n:n + 1].view(1, -1), cache)[0])           # single token
        parts.append(m(seq[n + 1:n + 6].view(1, -1), cache)[0])       # append-N
        inc = torch.cat(parts)
        assert torch.allclose(full[: n + 6], inc, atol=1e-4)
        # rollback 3, then re-append different-then-same tokens
        cache.truncate(cache.len - 3)
        m(torch.tensor([[tok.ent_tok(999), tok.ent_tok(998)]]), cache)
        cache.truncate(cache.len - 2)
        again = m(seq[n + 3:].view(1, -1), cache)[0]
        assert torch.allclose(full[n + 3:], again, atol=1e-4)


def test_expand(rand_model, data):
    tok, split = data
    m = rand_model
    ctx = torch.as_tensor(split["test"][1]["ctx"].tolist()).view(1, -1)
    with torch.no_grad():
        cache = m.new_cache(1, 1024)
        m(ctx, cache)
        cache.expand(3)
        nxt = torch.tensor([[tok.row_tok(1)], [tok.row_tok(2)], [tok.row_tok(3)]])
        out = m(nxt, cache)[:, -1]
        for b in range(3):
            ref = m(torch.cat([ctx[0], nxt[b]]).view(1, -1))[0, -1]
            assert torch.allclose(out[b], ref, atol=1e-4)
