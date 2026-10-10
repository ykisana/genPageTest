"""E8: adapter that poses the homepage as declared questions to TypeSafe's hosted Jev model.

Uses the official `typesafe_sdk` (System One: `system_one(state=..., questions=...)`), which reads
TYPESAFE_API_KEY from the environment. Nothing here makes a network call unless an SdkTransport
is constructed explicitly.

Question design (one System One call per page, mirroring the E7 local model):
  * rows come from GenPage's skeleton (row tokens), so Jev only answers entity questions;
  * per row: one `Choice` over the top-N GenPage-eligible candidates. Jev returns a probability
    for every option; ranking options by it gives the row's titles, the probability is the confidence.
Constraints (dedup across rows, eligibility, watched) are re-applied locally afterwards and the
number of answers that needed fixing is reported as `conflicts`.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from genpage.config import RESULTS_DIR
from genpage.decoding.constraints import Constraints, masked
from genpage.decoding.page import PageOut, Session
from genpage.eval.profile import NULL

CACHE_DIR = RESULTS_DIR / "jev_cache"


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class SdkTransport:
    """Live calls through typesafe_sdk. payload -> {"answers": {qid: {option_id: prob}}, "usage": {...}}."""

    def __init__(self, model=None, timeout=60.0):
        from typesafe_sdk import TypeSafeClient
        self.client = TypeSafeClient()
        self.model, self.timeout = model, timeout

    def __call__(self, payload: dict) -> dict:
        from typesafe_sdk import Choice
        questions = {q["id"]: Choice(instructions=q["prompt"], criteria={o["id"]: o["label"] for o in q["options"]})
                     for q in payload["questions"]}
        r = self.client.system_one(state=payload["context"], questions=questions, model=self.model,
                                   timeout=self.timeout)
        usage = r.usage.model_dump() if hasattr(r.usage, "model_dump") else {}
        return {"answers": {qid: dict(a.probabilities) for qid, a in r.choices.items()}, "usage": usage,
                "model": getattr(r, "model", None)}


class CachedTransport:
    """Content-addressed on-disk cache so reruns never re-bill."""

    def __init__(self, inner, cache_dir: Path = CACHE_DIR):
        self.inner, self.dir = inner, cache_dir

    def __call__(self, payload):
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        p = self.dir / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text())
        resp = self.inner(payload)
        self.dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(resp))
        return resp


class JevDecision:
    """Jev as the page's decision model.

    source:     "genpage" = GenPage picks the rows and scores candidates; "popular" = heuristic rows +
                popularity candidates (no learned model at all: the cold-start setting).
    dedup:      give every row its own non-overlapping candidate pool so parallel answers cannot collide.
    sequential: one call per row; each call sees the titles already placed above it (row-level
                autoregression). Otherwise one call answers every row in parallel.
    """

    def __init__(self, transport, genpage_model, tok, cat, R, L, n_candidates=50, hist_items=30,
                 source="genpage", dedup=False, sequential=False):
        self.t, self.m, self.tok, self.cat = transport, genpage_model, tok, cat
        self.R, self.L, self.N, self.H = R, L, n_candidates, hist_items
        self.source, self.dedup, self.sequential = source, dedup, sequential
        self.name = "jev" + ("-popular" if source == "popular" else "") + ("-dedup" if dedup else "") \
            + ("-seq" if sequential else "")
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    # --- request ---
    def describe_user(self, imp):
        ctx = imp["ctx"].tolist()
        hist = ctx[9:-1]
        items = [hist[i] - self.tok.ent0 for i in range(0, len(hist), 3)][-self.H:]
        rats = [hist[i + 1] - self.tok.rating0 + 1 for i in range(0, len(hist), 3)][-self.H:]
        return [f"{self.cat.titles[e]} — rated {r}/5" for e, r in zip(items, rats)]

    @torch.no_grad()
    def row_scores(self, imp, cons: Constraints):
        """Rows for the page and, per row, a score over all entities used to build candidate pools."""
        if self.source == "popular":
            from genpage.decoding.baselines import heuristic_rows
            rows = heuristic_rows(imp, self.tok, self.cat, cons, self.R)
            pop = torch.as_tensor(self.cat.popularity, device=cons.ct.dev)
            return rows, [pop] * len(rows)
        sess = Session(self.m, len(imp["ctx"]) + self.R + 2, prof=NULL)
        logits, _ = sess.fwd(imp["ctx"].tolist())
        cur = logits[-1]
        c = cons.clone()
        rows, scores = [], []
        for _ in range(self.R):
            row = int(masked(cur[self.tok.row_slice()], c.row_mask()).argmax())
            c.add_row(row)
            rows.append(row)
            logits, _ = sess.fwd([self.tok.row_tok(row)])
            cur = logits[-1]
            scores.append(cur[self.tok.ent_slice()].clone())
        return rows, scores

    def pool(self, scores, mask, exclude=()):
        m = mask.clone()
        if exclude:
            m[torch.as_tensor(sorted(exclude), device=m.device)] = False
        return masked(scores, m).topk(min(self.N, int(m.sum()))).indices.tolist()

    def round_robin_pools(self, rows, scores, c):
        """Disjoint per-row pools, allocated draft-style (each row in turn takes its next-best unclaimed
        title) so no row can hoard the titles other rows rank highly."""
        ranked = [masked(scores[r], c.ent_mask(row)).topk(min(self.N * len(rows), int(c.ent_mask(row).sum())))
                  .indices.tolist() for r, row in enumerate(rows)]
        pools, ptr, taken = [[] for _ in rows], [0] * len(rows), set()
        for _ in range(self.N):
            for r in range(len(rows)):
                while ptr[r] < len(ranked[r]) and ranked[r][ptr[r]] in taken:
                    ptr[r] += 1
                if ptr[r] < len(ranked[r]):
                    e = ranked[r][ptr[r]]
                    pools[r].append(e)
                    taken.add(e)
                    ptr[r] += 1
        return pools

    def label(self, e):
        from genpage.data.catalog import GENRES
        g = ", ".join(GENRES[i] for i in np.flatnonzero(self.cat.genres[e]))
        return f"{self.cat.titles[e]} [{g}]"

    def build_questions(self, imp, rows, cands, row_ids=None, placed=None):
        from genpage.data.catalog import ROW_NAMES
        context = dict(task="Personalized movie homepage for one user, built from their recent ratings "
                            "(most recent last).", user_history=self.describe_user(imp))
        if placed:
            context["already_on_page_above"] = [self.cat.titles[e] for e in placed]
        row_ids = row_ids if row_ids is not None else range(len(rows))
        return dict(
            context=context,
            questions=[dict(id=f"row{r}", type="ranked_choice", k=self.L,
                            prompt=f"Which of these titles is this user most likely to watch and enjoy next, "
                                   f"shown in a homepage row titled '{ROW_NAMES[row]}'?",
                            options=[dict(id=str(e), label=self.label(e)) for e in cand])
                       for r, row, cand in zip(row_ids, rows, cands)],
        )

    @staticmethod
    def parse_answers(resp, R, row_ids=None):
        """{"answers": {"row0": {"<entity id>": prob, ...}}} -> per row [(entity, prob)] ranked by prob."""
        out = []
        for r in (row_ids if row_ids is not None else range(R)):
            probs = resp["answers"][f"row{r}"]
            out.append([(int(k), float(p)) for k, p in sorted(probs.items(), key=lambda kv: -kv[1])])
        return out

    def ask(self, payload):
        resp = self.t(payload)
        u = resp.get("usage") or {}
        self.usage["input_tokens"] += int(u.get("input_tokens", 0) or 0)
        self.usage["output_tokens"] += int(u.get("output_tokens", 0) or 0)
        self.usage["calls"] += 1
        return resp

    def place(self, out, cons, row, ans, cand):
        """Apply constraints to one row's ranked answers (top-up from the pool if Jev under-answers)."""
        cons.add_row(row)
        ok = cons.ent_mask(row).cpu()
        pool = set(cand)
        valid = [(e, p) for e, p in ans if e in pool]
        out.invalid += len(ans) - len(valid)
        picked, vals = [], []
        for e, p in valid:
            if len(picked) == self.L:
                break
            if ok[e] and e not in picked:
                picked.append(e)
                vals.append(logit(p))
        for e in cand:
            if len(picked) == self.L:
                break
            if ok[e] and e not in picked:
                picked.append(e)
                vals.append(0.0)
        if len(picked) < self.L:                    # pool exhausted by dedup: fill from any eligible title
            for e in torch.nonzero(cons.ent_mask(row)).flatten().tolist():
                if len(picked) == self.L:
                    break
                if e not in picked:
                    picked.append(e)
                    vals.append(0.0)
        out.conflicts += len({e for e, _ in valid[: self.L]} - set(picked))
        cons.add_ents(picked)
        out.ents.append(picked)
        out.values.append(vals)

    # --- DecisionModel ---
    def decide_page(self, imp, cons: Constraints) -> PageOut:
        rows, scores = self.row_scores(imp, cons)
        out = PageOut(rows=rows, forwards=(0 if self.source == "popular" else 1 + self.R))
        out.conflicts = 0   # Jev top-L picks that had to be replaced (already used by an earlier row), as in E7
        out.invalid = 0     # answers outside the declared options
        out.calls = 0
        if self.sequential:
            for r, row in enumerate(rows):
                c = cons.clone()
                c.add_row(row)
                cand = self.pool(scores[r], c.ent_mask(row))
                placed = [e for es in out.ents for e in es]
                ans = self.parse_answers(self.ask(self.build_questions(imp, [row], [cand], [r], placed)), 1, [r])[0]
                out.calls += 1
                self.place(out, cons, row, ans, cand)
        else:
            c = cons.clone()
            for row in rows:
                c.add_row(row)
            if self.dedup:
                cands = self.round_robin_pools(rows, scores, c)
            else:
                cands = [self.pool(scores[r], c.ent_mask(row)) for r, row in enumerate(rows)]
            answers = self.parse_answers(self.ask(self.build_questions(imp, rows, cands)), len(rows))
            out.calls = 1
            for row, ans, cand in zip(rows, answers, cands):
                self.place(out, cons, row, ans, cand)
        out.k_used = [0] * self.R
        return out
