import torch

from rubik import tokenizer as T
from rubik.model import CubeLM, ModelConfig


def test_forward_and_padding_invariance():
    torch.manual_seed(0)
    m = CubeLM(ModelConfig(d_model=64, n_layers=2, n_heads=4)).eval()
    view = torch.randint(T.COLOR_BASE, T.COLOR_BASE + 6, (3, 27))
    hist = torch.randint(T.MOVE_BASE, T.SCRATCH_BASE, (3, 10))
    lens = torch.tensor([10, 4, 0])
    out = m(view, hist, lens)
    assert out.shape == (3, T.VOCAB)
    # padding content after hist_len must not matter
    hist2 = hist.clone()
    hist2[1, 4:] = 0
    hist2[2, :] = 0
    out2 = m(view, hist2, lens)
    assert torch.allclose(out, out2, atol=1e-5)
    # shorter pad: truncate columns
    out3 = m(view[2:], hist[2:, :0], lens[2:])
    assert torch.allclose(out[2:], out3, atol=1e-5)
