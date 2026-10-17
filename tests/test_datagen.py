import numpy as np
import pytest

from rubik import cube as C
from rubik import tokenizer as T
from rubik.data import gen


@pytest.mark.parametrize("kind", gen.KINDS)
def test_kind_replays(kind):
    cfg = gen.GenConfig(kind=kind, n=30, seed=3)
    d = gen.generate(cfg)
    st, tk, w, start = d["states"], d["tokens"], d["weights"], d["traj_start"]
    # recompute states from first state of each trajectory and compare
    i = 0
    n_traj = 0
    while i < len(tk):
        j = i
        s = st[i].copy()
        while j < len(tk) and start[j] == start[i]:
            assert np.array_equal(st[j], s), (kind, i, j)
            if T.is_cube_token(int(tk[j])):
                s = s[T.PERM_TABLE[int(tk[j])]]
            j += 1
        if kind in ("full", "pll", "rev"):
            assert tk[j - 1] == T.DONE
            assert C.is_solved(s)
        n_traj += 1
        i = j
    assert n_traj == 30


def test_inspection_covers_all_faces():
    s = C.apply_alg(C.SOLVED.copy(), gen.INSPECT)
    seen = set(int(c) for c in s[:27][[4, 13, 22]])
    assert seen == {3, 4, 5}
    s2 = C.apply_alg(s, gen.INSPECT_BACK)
    assert np.array_equal(s2, C.SOLVED)


def test_determinism():
    a = gen.generate(gen.GenConfig(kind="full", n=5, seed=11))
    b = gen.generate(gen.GenConfig(kind="full", n=5, seed=11))
    for k in a:
        assert np.array_equal(a[k], b[k])
