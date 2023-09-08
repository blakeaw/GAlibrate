import numpy as np
from galibrate.benchmarks import *

point = np.zeros(3)


def test_sphere():
    # Minimum: X_0 = (0, 0, ..., 0), f(X_0) = 0
    assert np.isclose(sphere(point), 0.0)


def test_cigar():
    # Minimum: X_0 = (0, 0, ..., 0), f(X_0) = 0
    assert np.isclose(cigar(point), 0.0)


def test_rastrigin():
    # Minimum: X_0 = (0, 0, ..., 0), f(X_0) = 0
    assert np.isclose(rastrigin(point), 0.0)


def test_himmelblau():
    # Minimum: X_1 = (3.0, 2.0), f(X_1) = 0
    min_1 = [3.0, 2.0]
    assert np.isclose(himmelblau(min_1[0], min_1[1]), 0.0)
    # Minimum: X_2 = (-2.805118, 3.131312), f(X_2) = 0
    min_2 = [-2.805118, 3.131312]
    assert np.isclose(himmelblau(min_2[0], min_2[1]), 0.0)
    # Minimum: X_3 = (-3.779310, -3.283186), f(X_3) = 0
    min_3 = [-3.779310, -3.283186]
    assert np.isclose(himmelblau(min_3[0], min_3[1]), 0.0)
    # Minimum: X_4 = (3.584428, -1.848126), f(X_4) = 0
    min_4 = [3.584428, -1.848126]
    assert np.isclose(himmelblau(min_4[0], min_4[1]), 0.0)


if __name__ == "__main__":
    test_cigar()
    test_cigar()
    test_rastrigin()
    test_himmelblau()
