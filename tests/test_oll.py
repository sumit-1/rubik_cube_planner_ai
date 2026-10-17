import time

import numpy as np

from rubik import cube as C
from rubik.teacher import oll


def _random_ll_state(rng):
    """Solved cube + random LL scramble (via random U/R/F algs that keep F2L): use inverse of
    a random <R,U,F> walk, then check F2L solved by construction (scramble only LL)."""
    # random LL state: apply random sexy-type macros that preserve F2L
    s = C.SOLVED.copy()
    algs = ["R U R' U R U2 R'", "R U2 R' U' R U' R'", "F R U R' U' F'",
            "R U R' U' R' F R F'", "R U R' U R U' R' U R U2 R'", "U", "U'", "U2"]
    for _ in range(rng.integers(3, 9)):
        s = C.apply_alg(s, algs[rng.integers(len(algs))])
    return s


def test_oll_cases():
    rng = np.random.default_rng(0)
    t0 = time.time()
    keys = set()
    for _ in range(40):
        s = _random_ll_state(rng)
        alg = oll.solve_oll(s)
        s2 = C.apply_alg(s, alg)
        assert oll.oll_oriented(s2)
        # F2L untouched
        for p in ("DF", "DR", "DB", "DL", "FR", "FL", "BL", "BR", "DFR", "DFL", "DBL", "DBR"):
            for st in C.piece_of(p):
                assert s2[st] == C.SOLVED[st]
        keys.add(oll.oll_case_key(s))
    print("cases", len(keys), "time", time.time() - t0)
