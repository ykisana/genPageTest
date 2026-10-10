"""Non-learned baselines: heuristic row choice + popularity ranking (the cold-start reference)."""
import numpy as np
import torch

from genpage.decoding.constraints import Constraints, masked
from genpage.decoding.page import PageOut


def history_items(imp, tok):
    hist = imp["ctx"].tolist()[9:-1]
    items = np.array([hist[i] - tok.ent0 for i in range(0, len(hist), 3)], dtype=np.int64)
    rats = np.array([hist[i + 1] - tok.rating0 + 1 for i in range(0, len(hist), 3)], dtype=np.int64)
    return items, rats


def heuristic_rows(imp, tok, cat, cons: Constraints, R):
    """Top Picks, then the rows whose eligible titles the user rated >= 4 most often (feasible only)."""
    items, rats = history_items(imp, tok)
    liked = items[rats >= 4] if (rats >= 4).any() else items
    affinity = cat.eligibility[:, liked].sum(1).astype(float)
    affinity[0] = np.inf                      # Top Picks is pinned first
    c = cons.clone()
    rows = []
    for t in np.argsort(-affinity, kind="stable"):
        if len(rows) == R:
            break
        if bool(c.row_mask()[int(t)]):
            c.add_row(int(t))
            rows.append(int(t))
    return rows


class PopularityBaseline:
    name = "popularity"

    def __init__(self, tok, cat, R, L, dev):
        self.tok, self.cat, self.R, self.L = tok, cat, R, L
        self.pop = torch.as_tensor(cat.popularity, device=dev)

    def decide_page(self, imp, cons: Constraints) -> PageOut:
        rows = heuristic_rows(imp, self.tok, self.cat, cons, self.R)
        out = PageOut(rows=rows, forwards=0)
        for row in rows:
            cons.add_row(row)
            top = masked(self.pop, cons.ent_mask(row)).topk(self.L).indices.tolist()
            cons.add_ents(top)
            out.ents.append(top)
            out.values.append([0.0] * self.L)
        out.k_used = [0] * self.R
        return out
