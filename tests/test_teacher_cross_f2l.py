import time

import numpy as np

from rubik import cube as C
from rubik.teacher import cross, f2l


def _scrambled(rng, n=25):
    return C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, n))


def test_cross_solves():
    rng = np.random.default_rng(0)
    lens = []
    for _ in range(300):
        s = _scrambled(rng)
        mv = cross.solve_cross(s)
        s2 = C.apply_alg(s, mv)
        assert cross.cross_solved(s2)
        # direct check
        for p in ("DF", "DR", "DB", "DL"):
            for st in C.piece_of(p):
                assert s2[st] == C.SOLVED[st]
        lens.append(len(mv))
    assert max(lens) <= 8


def test_f2l_solves():
    rng = np.random.default_rng(1)
    t0 = time.time()
    lens = []
    for _ in range(50):
        s = _scrambled(rng)
        s = C.apply_alg(s, cross.solve_cross(s))
        for a, mv in f2l.solve_f2l(s):
            s = C.apply_alg(s, mv)
            lens.append(len(mv))
        # bottom two layers solved
        for p in ("DF", "DR", "DB", "DL", "FR", "FL", "BL", "BR", "DFR", "DFL", "DBL", "DBR"):
            for st in C.piece_of(p):
                assert s[st] == C.SOLVED[st], p
    print("avg pair len", np.mean(lens), "max", max(lens), "time", time.time() - t0)
