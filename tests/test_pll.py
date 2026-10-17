import numpy as np

from rubik import cube as C
from rubik.teacher import pll


def test_library_verified():
    print("verified PLL algs:", sorted(pll.LIBRARY), "missing:",
          sorted(set(pll.PLL_ALGS) - set(pll.LIBRARY)))
    assert len(pll.LIBRARY) >= 8


def test_rotation_free_conversion():
    rng = np.random.default_rng(0)
    for name in ("Aa", "V", "E"):
        alg = pll.PLL_ALGS[name]
        seq = pll.to_face_turns(alg)
        assert not any(t[0] in "xyz" for t in seq)
        s = C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, 15))
        a = C.canonicalize(C.apply_alg(s, alg))[0]
        b = C.apply_alg(s, seq)
        assert np.array_equal(a, b), name


def test_all_ll_perms_solvable():
    # enumerate PLL states: U^a * alg * U^b for all library algs
    seen = {}
    for seq in list(pll.LIBRARY.values()) + [[]]:
        for pre in range(4):
            for post in range(4):
                s = C.apply_alg(C.SOLVED.copy(), ["U"] * pre + seq + ["U"] * post)
                seen[s.tobytes()] = s
    print("distinct PLL states", len(seen))
    assert len(seen) == 288
    lens = []
    for s in seen.values():
        mv = pll.solve_pll(s)
        s2 = C.apply_alg(s, mv)
        assert np.array_equal(s2, C.SOLVED)
        lens.append(len(mv))
    print("PLL avg len", np.mean(lens), "max", max(lens))
