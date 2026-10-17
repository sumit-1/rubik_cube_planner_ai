import time

import numpy as np

from rubik import cube as C
from rubik.teacher import cfop


def test_full_cfop_10k():
    rng = np.random.default_rng(123)
    n = 10000
    t0 = time.time()
    lens = []
    for i in range(n):
        s = C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, 25))
        plan = cfop.full_plan(s)
        s2 = C.apply_alg(s, [m for _, m in plan])
        assert np.array_equal(s2, C.SOLVED), i
        lens.append(len(plan))
    print(f"{n} solves in {time.time() - t0:.1f}s avg len {np.mean(lens):.1f} max {max(lens)}")


def test_frame_conversion():
    rng = np.random.default_rng(5)
    for _ in range(100):
        s = C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, 20))
        plan = cfop.full_plan(s)[:12]
        rot = [str(rng.choice(C._ROT_BASE)) for _ in range(rng.integers(1, 4))]
        cur = C.apply_alg(s, rot)
        canon = s
        for _, m in plan:
            tok = cfop.to_current_token(m, cur)
            cur = C.apply_alg(cur, tok)
            canon = C.apply_alg(canon, m)
            assert np.array_equal(C.canonicalize(cur)[0], canon)
