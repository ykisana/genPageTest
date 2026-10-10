"""Expected calibration error + reliability bins for stated confidences vs. realized outcomes."""
import numpy as np


def reliability(conf, label, n_bins=15):
    conf, label = np.asarray(conf, float), np.asarray(label, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append(dict(lo=edges[b], hi=edges[b + 1], n=int(m.sum()),
                             conf=float(conf[m].mean()), acc=float(label[m].mean())))
    return bins


def ece(conf, label, n_bins=15) -> float:
    bins = reliability(conf, label, n_bins)
    n = sum(b["n"] for b in bins)
    return float(sum(b["n"] / n * abs(b["conf"] - b["acc"]) for b in bins))
