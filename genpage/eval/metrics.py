"""Offline page metrics, computed against each impression's held-out future ratings."""
import numpy as np

from genpage.data.catalog import Catalog
from genpage.data.teacher import MISS_REWARD, REWARD_BY_RATING


def row_disc(r):
    return 1.0 / np.log2(r + 2)


def violations(rows, ents, imp, cat: Catalog, L: int) -> int:
    """Independent re-check of every business rule (does not reuse the decoder's masks)."""
    v = 0
    flat = [e for row in ents for e in row]
    v += len(flat) - len(set(flat))                        # duplicate entities
    v += len(rows) - len(set(rows))                        # duplicate rows
    v += int(rows[0] != 0)                                 # Top Picks pinned at row 0
    v += int(0 in rows[1:])
    watched = set(imp["watched"].tolist())
    for row, es in zip(rows, ents):
        v += abs(len(es) - L)
        v += sum(not cat.eligibility[row, e] for e in es)
        v += sum(e in watched for e in es)
    return v


def page_metrics(rows, ents, imp, cat: Catalog) -> dict:
    fut = dict(zip(imp["fut_items"].tolist(), imp["fut_ratings"].tolist()))
    pos = {i for i, r in fut.items() if r >= 4}
    gain, reward, hits = 0.0, 0.0, set()
    for r, es in enumerate(ents):
        for c, e in enumerate(es):
            rating = fut.get(e)
            reward += REWARD_BY_RATING[rating] if rating else MISS_REWARD
            if rating and rating >= 4:
                gain += (rating - 3) * row_disc(r) * row_disc(c)
                hits.add(e)
    flat = np.array([e for es in ents for e in es])
    g = cat.genres[flat]
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    sim = g @ g.T
    n = len(flat)
    diversity = 1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return dict(gain=gain, reward=reward, recall=len(hits) / len(pos) if pos else np.nan, diversity=diversity)


def agreement(a_ents, b_ents) -> dict:
    """Slot-level and page-level exact agreement between two decoded pages."""
    slots = sum(x == y for ra, rb in zip(a_ents, b_ents) for x, y in zip(ra, rb))
    total = sum(len(r) for r in a_ents)
    return dict(slot_agree=slots / total, page_exact=float(a_ents == b_ents))


def summarize(records: list) -> dict:
    keys = records[0].keys()
    return {k: float(np.nanmean([r[k] for r in records])) for k in keys}
