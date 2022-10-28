"""
This module defines several functions used to test and benchmark
single objective opitimization algorithms.

References:
 1. https://deap.readthedocs.io/en/master/api/benchmarks.html
 2. https://en.m.wikipedia.org/wiki/Test_functions_for_optimization
 3. https://www.sfu.ca/~ssurjano/optimization.html
 4. http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm
"""

import numpy as np


def sphere(X: np.array) -> float:
    """Returns the value of an N-dimensional sphere function.
    The sphere function is given by,
        f(X) = sum_(i=1)**N x_i**2 ,
    and is defined on range,
        [-inf, inf] ,
    with a minimum at,
        X_0 = (0, 0, ..., 0), f(X_0) = 0 .

    Args:
        X: Input position vector.
    """
    return np.sum(X**2)

def cigar(X: np.array) -> float:
    """Returns the value of an N-dimensional Cigar function.
    The cigar function is given by,
        f(X) = x_1**2 + 10**6 * sum_(i=2)^N x_i**2 ,
    and is defined on range,
        [-inf, inf] ,
    with a minimum at,
        X_0 = (0, 0, ..., 0), f(X_0) = 0 .

    Args:
        X: Input position vector.
    """
    return X[0]**2 + 1e6*np.sum(X[1:]**2)


def rastrigin(X: np.array) -> float:
    """Returns the value of the N-dimensional Rastrigin function.
    The Rastrigin function is given by,
        f(X) = 10*N + sum_(i=1)^N x_i**2 - 10*cos(2*pi*x_i) ,
    and is defined on range,
        [-5.12, 5.12] ,
    with a minimum at,
        X_0 = (0, 0, ..., 0), f(X_0) = 0
    """
    N = len(X)
    xsq = X**2
    cos2pix = np.cos(2*np.pi*X)
    return 10*N + np.sum(xsq - 10*cos2pix)

# Define the fitness function (minimize himmelblau function)
# minima:
# x1 = (3.0,2.0) and f(x1) = 0
# x2 = (-2.805118,3.131312) and f(x2) = 0
# x3 = (-3.779310,-3.283186) and f(x3) = 0
# x4 = (3.584428,-1.848126) and f(x4) = 0
def himmelblau(x_1: float, x_2: float) -> float:
    """Returns the value of the 2-dimensional Himmelblau function.

    """
    return (x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2 - 7)**2
