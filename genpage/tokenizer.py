"""Domain-specific tokenizer: one token per row, one per entity, bucketized context signals.

Sequence layout (mirrors GenPage Fig. 1):
  <BOS> <PROFILE> gender age occ <REQ> hour wday <HIST> (ent rating dt)*N <PAGE> (row e1..eL)*R <EOS>
"""
import datetime as dt

import numpy as np

from genpage.data.catalog import AGES, N_OCCUPATIONS, ROW_NAMES

SPECIALS = ["<PAD>", "<BOS>", "<PROFILE>", "<REQ>", "<HIST>", "<PAGE>", "<EOS>", "<UNK_ROW>", "<UNK_ENT>"]
# Age of an action relative to the impression, in seconds.
DT_EDGES = [60, 3600, 86400, 7 * 86400, 30 * 86400, 182 * 86400, 365 * 86400]
N_DT = len(DT_EDGES) + 1
N_HOUR = 6   # 4-hour buckets
N_WDAY = 7


class Tokenizer:
    def __init__(self, n_entities: int):
        self.n_entities = n_entities
        self.n_row_types = len(ROW_NAMES)
        off = 0
        self.special = {s: i for i, s in enumerate(SPECIALS)}
        off += len(SPECIALS)
        self.rating0 = off; off += 5
        self.dt0 = off; off += N_DT
        self.gender0 = off; off += 2
        self.age0 = off; off += len(AGES)
        self.occ0 = off; off += N_OCCUPATIONS
        self.hour0 = off; off += N_HOUR
        self.wday0 = off; off += N_WDAY
        self.row0 = off; off += self.n_row_types
        self.ent0 = off; off += n_entities
        self.vocab_size = off

    # --- ids ---
    def __getattr__(self, name):
        # tok.PAD, tok.BOS, ... for the special tokens
        if name.isupper() and f"<{name}>" in SPECIALS:
            return SPECIALS.index(f"<{name}>")
        raise AttributeError(name)

    def row_tok(self, r):
        return self.row0 + r

    def ent_tok(self, e):
        return self.ent0 + e

    def is_row(self, t):
        return self.row0 <= t < self.row0 + self.n_row_types

    def is_ent(self, t):
        return self.ent0 <= t < self.ent0 + self.n_entities

    def row_slice(self):
        return slice(self.row0, self.row0 + self.n_row_types)

    def ent_slice(self):
        return slice(self.ent0, self.ent0 + self.n_entities)

    # --- encoding ---
    @staticmethod
    def dt_bucket(seconds) -> np.ndarray:
        return np.searchsorted(DT_EDGES, np.asarray(seconds), side="right")

    def encode_context(self, gender: str, age: int, occ: int, ts: int,
                       hist_items: np.ndarray, hist_ratings: np.ndarray, hist_ts: np.ndarray) -> list:
        t = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
        toks = [self.BOS, self.PROFILE,
                self.gender0 + (0 if gender == "F" else 1),
                self.age0 + AGES.index(int(age)),
                self.occ0 + int(occ),
                self.REQ, self.hour0 + t.hour // 4, self.wday0 + t.weekday(),
                self.HIST]
        if len(hist_items):
            trip = np.stack([
                self.ent0 + np.asarray(hist_items),
                self.rating0 + np.asarray(hist_ratings) - 1,
                self.dt0 + self.dt_bucket(ts - np.asarray(hist_ts)),
            ], axis=1)
            toks.extend(trip.reshape(-1).tolist())
        toks.append(self.PAGE)
        return toks

    def encode_page(self, rows, ents) -> list:
        """rows: [R] row-type ids, ents: [R, L] entity ids -> page tokens incl. <EOS>."""
        out = []
        for r, row_ents in zip(rows, ents):
            out.append(self.row_tok(int(r)))
            out.extend(self.ent_tok(int(e)) for e in row_ents)
        out.append(self.EOS)
        return out

    def decode_page(self, toks):
        """Inverse of encode_page (stops at <EOS>). Returns (rows, ents)."""
        rows, ents = [], []
        for t in toks:
            if t == self.EOS:
                break
            if self.is_row(t):
                rows.append(t - self.row0)
                ents.append([])
            elif self.is_ent(t):
                if not ents:
                    raise ValueError("entity before any row")
                ents[-1].append(t - self.ent0)
            else:
                raise ValueError(f"unexpected token {t} in page")
        return rows, ents
