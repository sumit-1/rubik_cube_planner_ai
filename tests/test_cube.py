import numpy as np
import pytest

from rubik import cube as C


def test_perms_are_permutations():
    for k, p in C.PERM.items():
        assert sorted(p.tolist()) == list(range(54)), k


@pytest.mark.parametrize("m", C.CUBE_ACTIONS)
def test_inverse_and_order(m):
    s = C.SOLVED.copy()
    s2 = C.apply_alg(s, [m, C.invert_move(m)])
    assert np.array_equal(s2, s)
    reps = 2 if m.endswith("2") else 4
    assert np.array_equal(C.apply_alg(s, [m] * reps), s)
    assert not np.array_equal(C.apply_alg(s, [m]), s)


def test_sexy_move_period():
    s = C.SOLVED.copy()
    assert np.array_equal(C.apply_alg(s, "R U R' U'" * 1), C.apply_alg(s, "R U R' U'"))
    t = s
    for _ in range(6):
        t = C.apply_alg(t, "R U R' U'")
    assert np.array_equal(t, s)
    t = s
    for _ in range(5):
        t = C.apply_alg(t, "R U R' U'")
    assert not np.array_equal(t, s)


def test_U_turn_effect():
    s = C.apply_alg(C.SOLVED.copy(), "U")
    # clockwise U: F face top row takes colour of R face (R -> F)
    assert list(s[18:21]) == [1, 1, 1]
    assert list(s[9:12]) == [5, 5, 5]  # R top row <- B
    # centers unchanged
    assert list(s[C.CENTER_IDX]) == [0, 1, 2, 3, 4, 5]


def test_rotation_y_moves_centers_and_view():
    s = C.apply_alg(C.SOLVED.copy(), "y")
    # after y, the front face shows what used to be the right face (U-like rotation)
    assert s[22] == 1 and s[13] == 5 and s[4] == 0
    v = C.view_ufr(C.apply_alg(C.SOLVED.copy(), "x2"))
    assert v[4] == 3  # U-face center now D color


def test_rotation_commutes_with_conjugated_move():
    rng = np.random.default_rng(0)
    s = C.apply_alg(C.SOLVED.copy(), C.random_scramble(rng, 20))
    # y then U  ==  (turn the face that is now U) i.e. U after y == U before y
    a = C.apply_alg(s, "y U")
    b = C.apply_alg(s, "U y")
    assert np.array_equal(a, b)
    # y then F == R then y' ... check: turning F after y equals turning R before y
    assert np.array_equal(C.apply_alg(s, "y F"), C.apply_alg(s, "R y"))


def test_slice_and_wide_moves():
    s = C.SOLVED.copy()
    assert np.array_equal(C.apply_alg(s, "M M' E E' S S'"), s)
    assert np.array_equal(C.apply_alg(s, "r"), C.apply_alg(s, "R M'"))
    # M2 U M2 U2 M2 U M2 is H perm: edges swapped, corners solved
    h = C.apply_alg(s, "M2 U M2 U2 M2 U M2")
    assert not np.array_equal(h, s)
    assert np.array_equal(h[C.CORNER_STICKERS], s[C.CORNER_STICKERS])  # H perm: corners fixed
    assert np.array_equal(C.apply_alg(h, "M2 U M2 U2 M2 U M2"), s)


def test_canonicalize():
    rng = np.random.default_rng(1)
    for _ in range(50):
        scr = C.random_scramble(rng, 15)
        base = C.apply_alg(C.SOLVED.copy(), scr)
        rot = [str(rng.choice(C._ROT_BASE)) for _ in range(rng.integers(0, 5))]
        s = C.apply_alg(base, rot)
        canon, seq = C.canonicalize(s)
        assert list(canon[C.CENTER_IDX]) == [0, 1, 2, 3, 4, 5]
        assert np.array_equal(canon, C.apply_alg(s, seq))
        # canonical state equals base state (rotations only relabel)
        assert np.array_equal(canon, base)


def test_pieces():
    assert len(C.piece_of("DF")) == 2
    assert len(C.piece_of("DFR")) == 3
    assert C.sticker_of("DF", "D") in C.piece_of("DF")
