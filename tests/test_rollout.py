import numpy as np
import torch

from rubik import cube as C
from rubik import tokenizer as T
from rubik.gpucube import BatchCube
from rubik.model import CubeLM, ModelConfig
from rubik.rollout import rollout


def test_gpu_env_matches_numpy():
    env = BatchCube("cuda")
    rng = np.random.default_rng(0)
    toks = rng.choice(T.CUBE_IDS, size=(8, 30))
    st = env.solved_batch(8)
    states = env.run(st, torch.as_tensor(toks, device="cuda"))
    for b in range(8):
        s = C.SOLVED.copy()
        for t in toks[b]:
            s = s[T.PERM_TABLE[t]]
        assert np.array_equal(states[b, -1].cpu().numpy(), s)


def test_rollout_recording_consistent():
    torch.manual_seed(0)
    model = CubeLM(ModelConfig(d_model=64, n_layers=2, n_heads=4)).cuda()
    env = BatchCube("cuda")
    rng = np.random.default_rng(1)
    start = np.stack([C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, 6)) for _ in range(6)]).astype(np.int64)
    out = rollout(model, env, torch.as_tensor(start, device="cuda"), 12, greedy=False, allow_scratch=True,
                  record=True)
    rs, rt, ra = out["rec_states"], out["rec_tokens"], out["rec_alive"]
    for b in range(6):
        s = start[b].copy()
        for t in range(rt.shape[1]):
            if not ra[b, t]:
                break
            assert np.array_equal(rs[b, t].cpu().numpy(), s)
            s = s[T.PERM_TABLE[int(rt[b, t])]]
        assert np.array_equal(out["state"][b].cpu().numpy(), s)
