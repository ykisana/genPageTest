"""Configuration dataclasses shared by data building, training, and decoding."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ML_DIR = DATA_DIR / "ml-1m"
PROC_DIR = DATA_DIR / "processed"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


@dataclass
class DataConfig:
    n_rows: int = 6               # rows per page (R)
    row_len: int = 10             # entities per row (L)
    hist_cap: int = 100           # most recent actions kept in the prompt
    future_window: int = 30       # next N ratings used as feedback / eval ground truth
    impression_every: int = 20    # one impression every N ratings per user
    min_history: int = 10
    min_future: int = 5
    test_frac: float = 0.10
    val_frac: float = 0.05
    popular_k: int = 500          # "Popular" row is restricted to the top-K most-rated titles
    mmr_lambda: float = 1.0       # teacher intra-row diversity strength (the controlled variable)
    gumbel_tau: float = 0.05      # teacher exploration noise
    row_tau: float = 0.5          # teacher noise on row choice
    svd_dim: int = 32
    seed: int = 0

    def tag(self) -> str:
        return f"lam{self.mmr_lambda:g}_h{self.hist_cap}"


@dataclass
class ModelConfig:
    vocab_size: int = 0
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    max_len: int = 2048
    dropout: float = 0.0
    n_mtp_heads: int = 0          # Medusa-style draft heads (E4)
    n_row_slots: int = 0          # >0 enables the row-index embedding used by the skeleton model (E5)

    @staticmethod
    def preset(name: str, vocab_size: int, **kw) -> "ModelConfig":
        sizes = {
            "tiny": dict(d_model=64, n_layers=2, n_heads=2),
            "S": dict(d_model=256, n_layers=4, n_heads=4),
            "M": dict(d_model=512, n_layers=8, n_heads=8),
        }
        return ModelConfig(vocab_size=vocab_size, **{**sizes[name], **kw})


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    n_random_neg: int = 16        # WBC random negatives per target
    neg_weight: float = 0.1
    max_steps: int = 0            # 0 = no cap (used by --tiny smoke runs)
    log_every: int = 100
    seed: int = 0


def as_dict(cfg) -> dict:
    return asdict(cfg)
