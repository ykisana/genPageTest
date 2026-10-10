"""Shared interface for Jev-style decision models: given a request, answer a fixed set of declared
questions (one choice per row slot and per entity slot) in parallel, each with a confidence.
Both the local parallel model (E7) and the hosted Jev API adapter (E8) implement it, so the same
harness evaluates either."""
from typing import Protocol

from genpage.decoding.constraints import Constraints
from genpage.decoding.page import PageOut


class DecisionModel(Protocol):
    name: str

    def decide_page(self, imp: dict, cons: Constraints) -> PageOut:
        """Return a full page. PageOut.values hold the stated confidence per slot as a logit."""
        ...


class LocalParallelDecision:
    def __init__(self, model, tok, R, L, joint: bool):
        self.model, self.tok, self.R, self.L, self.joint = model, tok, R, L, joint
        self.name = f"parallel-{'joint' if joint else 'indep'}"

    def decide_page(self, imp, cons):
        from genpage.decoding.parallel_fill import decode_parallel
        return decode_parallel(self.model, self.tok, imp["ctx"].tolist(), cons, self.R, self.L, self.joint)
