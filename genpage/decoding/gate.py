"""E6 decision model: a calibrated yes/no "is the current ranking stable?" gate.

Trained on validation impressions: at each AR step j>=1 we record features of the masked entity
ranking and whether stopping there (taking the tail from these logits) reproduces the full-k row
exactly. Logistic regression gives a calibrated P(stopping now is lossless).
"""
import numpy as np
import torch

from genpage.decoding.page import decode_page


def collect(model, tok, imps, make_cons, R, L, k):
    X, y = [], []
    for imp in imps:
        out = decode_page(model, tok, imp["ctx"].tolist(), make_cons(imp), R, L, k, fill="ar", record_gate=True)
        for feats, label, _, _ in out.gate_records:
            X.append(feats)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class Gate:
    def __init__(self, w, b, mu, sd, dev):
        self.w = torch.as_tensor(w, device=dev)
        self.b = float(b)
        self.mu = torch.as_tensor(mu, device=dev)
        self.sd = torch.as_tensor(sd, device=dev)

    def prob_np(self, X):
        z = ((X - self.mu.cpu().numpy()) / self.sd.cpu().numpy()) @ self.w.cpu().numpy() + self.b
        return 1 / (1 + np.exp(-z))

    def __call__(self, feats: torch.Tensor) -> float:
        z = ((feats - self.mu) / self.sd) @ self.w + self.b
        return float(torch.sigmoid(z))

    @staticmethod
    def fit(X, y, dev, l2=1e-3, iters=500):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.from_numpy((X - mu) / sd)
        yt = torch.from_numpy(y)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * w.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return Gate(w.detach().numpy(), b.item(), mu, sd, dev)
