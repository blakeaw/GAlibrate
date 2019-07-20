# GAlibrate

![Python version badge](https://img.shields.io/badge/python-3.6,3.7-blue.svg)
[![license](https://img.shields.io/github/license/blakeaw/GAlibrate.svg)](LICENSE)
![version](https://img.shields.io/badge/version-0.1.0-orange.svg)

GAlibrate is a python toolkit that provides an easy to use interface for model calibration/parameter estimation using an implementation of continuous genetic algorithm-based optimization.

GAlibrate's API was designed to be familiar to users of [PyDREAM](https://github.com/LoLab-VU/PyDREAM), [SimplePSO](https://github.com/LoLab-VU/ParticleSwarmOptimization), and [Gleipnir](https://github.com/LoLab-VU/Gleipnir), which are primarily used for biological model calibration.

------

# Install

| **! Warning** |
| :--- |
|  GAlibrate is still under heavy development and may rapidly change. |

GAlibrate installs as the `galibrate` package. It is compatible (i.e., tested) with Python 3.6 and 3.7.

Note that `galibrate` has the following core dependencies:
   * [NumPy](http://www.numpy.org/)
   * [SciPy](https://www.scipy.org/)

#### pip install
You can install the `galibrate` package using `pip` sourced from the GitHub repo:
```
pip install -e git+https://github.com/blakeaw/GAlibrate#egg=galibrate
```
However, this will not automatically install the core dependencies. You will have to do that separately:
```
pip install numpy scipy
```

### Recommended additional software

The following software is not required for the basic operation of GAlibrate, but provides extra capabilities and features when installed.

#### Cython
GAlibrate includes an implementation of the core genetic algorithm that is written in   [Cython](https://cython.org/), which takes advantage of Cython-based optimizations and compilation to accelerate the algorithm. This version of genetic algorithm is used if Cython is installed.

#### Numba
GAlibrate also includes an implementation of the core genetic algorithm that takes advantage of [Numba](https://numba.pydata.org/)-based JIT compilation and optimization to accelerate the algorithm. This version of genetic algorithm is used if Numba is installed.

------

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

------

# Documentation and Usage

### Quick Overview
Principally, **GAlibrate** defines the GAO (continuous **G**enetic **A**lgorithm-based **O**ptimizer ) class,
```python
from galibrate import GAO
```
which defines an object that can be used setup and run a continuous genetic algorithm-based optimization (i.e., a maximization) of a user-defined fitness function over the search space of a given set of (model) parameters.  

### Examples
Checkout the [examples](./examples) to see example scripts that show how to
setup and launch Genetic Algorithm runs using GAlibrate.

------

# Contact

To report problems or bugs please open a
[GitHub Issue](https://github.com/blakeaw/GAlibrate/issues). Additionally, any
comments, suggestions, or feature requests for GAlibrate can also be submitted as a
[GitHub Issue](https://github.com/blakeaw/GAlibrate/issues).

------

# Citing

If you use the GAlibrate software in your research, please cite the GitHub repo.
