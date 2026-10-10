import pytest
import torch

from genpage.config import DataConfig, ModelConfig
from genpage.data.catalog import load_catalog
from genpage.model import GenPageLM
from genpage.train.core import load_split


@pytest.fixture(scope="session")
def data():
    tok, split = load_split(DataConfig(mmr_lambda=1.0), tiny=True)
    return tok, split


@pytest.fixture(scope="session")
def cat():
    return load_catalog()


@pytest.fixture(scope="session")
def rand_model(data):
    tok, _ = data
    torch.manual_seed(0)
    m = GenPageLM(ModelConfig.preset("tiny", tok.vocab_size, max_len=1024, n_mtp_heads=2)).eval()
    # random init gives tiny logit gaps; scale the head up so rankings are not all near-ties
    with torch.no_grad():
        m.head.weight.mul_(50)
        for h in m.mtp:
            h.out.weight.normal_(0, 1)
    return m
