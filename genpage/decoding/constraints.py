"""Business-rule constraints as per-request, per-step token masks (GenPage Sec. 6.3).

Rules: Top Picks pinned at row 0; no repeated rows; a row is only eligible while it can still
be filled; entities must be eligible for their row (genre/decade/popular/highly-rated);
no duplicate entities on the page; no already-watched titles.
"""
import numpy as np
import torch

from genpage.data.catalog import Catalog

NEG_INF = float("-inf")


class CatalogTensors:
    """Per-device static tensors shared by all requests."""

    def __init__(self, cat: Catalog, dev):
        self.elig = torch.from_numpy(cat.eligibility).to(dev)            # [NR, NE]
        self.n_rows, self.n_ents = self.elig.shape
        self.dev = dev


class Constraints:
    def __init__(self, ct: CatalogTensors, watched: np.ndarray, row_len: int):
        self.ct, self.L = ct, row_len
        self.avail = torch.ones(ct.n_ents, dtype=torch.bool, device=ct.dev)
        if len(watched):
            self.avail[torch.as_tensor(watched, dtype=torch.long, device=ct.dev)] = False
        self.row_free = torch.ones(ct.n_rows, dtype=torch.bool, device=ct.dev)
        self.n_rows_placed = 0

    def clone(self) -> "Constraints":
        c = Constraints.__new__(Constraints)
        c.ct, c.L, c.n_rows_placed = self.ct, self.L, self.n_rows_placed
        c.avail, c.row_free = self.avail.clone(), self.row_free.clone()
        return c

    # --- masks ---
    def row_mask(self) -> torch.Tensor:
        if self.n_rows_placed == 0:
            m = torch.zeros_like(self.row_free)
            m[0] = True
            return m
        feasible = (self.ct.elig & self.avail).sum(1) >= self.L
        m = self.row_free & feasible
        m[0] = False
        return m

    def ent_mask(self, row: int) -> torch.Tensor:
        return self.ct.elig[row] & self.avail

    # --- updates ---
    def add_row(self, row: int):
        self.row_free[row] = False
        self.n_rows_placed += 1

    def add_ents(self, ents):
        self.avail[torch.as_tensor(ents, dtype=torch.long, device=self.ct.dev)] = False


def masked(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~mask, NEG_INF)
