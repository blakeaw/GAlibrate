# GAlibrate

![Python version badge](https://img.shields.io/badge/python-3.6,3.7-blue.svg)
[![license](https://img.shields.io/github/license/blakeaw/GAlibrate.svg)](LICENSE)
![version](https://img.shields.io/badge/version-0.4.0-orange.svg)
[![release](https://img.shields.io/github/release-pre/blakeaw/GAlibrate.svg)](https://github.com/blakeaw/GAlibrate/releases/tag/v0.4.0)
[![anaconda cloud](https://anaconda.org/blakeaw/galibrate/badges/version.svg)](https://anaconda.org/blakeaw/galibrate)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/6cdd91c06b11458384becb85db9adb15)](https://www.codacy.com/app/blakeaw1102/GAlibrate?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=blakeaw/GAlibrate&amp;utm_campaign=Badge_Grade)
[![DOI](https://zenodo.org/badge/197295657.svg)](https://zenodo.org/badge/latestdoi/197295657)

<p align="center">
  <img width="200" height="200" src="./images/GAlibrate_logo.png">
</p>

**GAlibrate** is a python toolkit that provides an easy to use interface for model calibration/parameter estimation using an implementation of continuous genetic algorithm-based optimization. Its functionality and API were designed to be familiar to users of the [PyDREAM](https://github.com/LoLab-VU/PyDREAM), [simplePSO](https://github.com/LoLab-VU/ParticleSwarmOptimization), and [Gleipnir](https://github.com/LoLab-VU/Gleipnir) packages.

Although **GAlibrate** provides a general framework for running continuous
genetic algorithm-based optimizations, it was created with systems biology models in mind. It therefore supplies additional tools for working with biological models in the [PySB](http://pysb.org/) format.

------

# Install

| **! Warning** |
| :--- |
|  GAlibrate is still under heavy development and may rapidly change. |

**GAlibrate** installs as the `galibrate` package. It is compatible (i.e., tested) with Python 3.6 and 3.7.

Note that `galibrate` has the following core dependencies:
   * [NumPy](http://www.numpy.org/)
   * [SciPy](https://www.scipy.org/)

#### pip install
You can install the latest release of the `galibrate` package using `pip` sourced from the GitHub repo:
```
pip install -e git+https://github.com/blakeaw/GAlibrate@v0.4.0#egg=galibrate
```
However, this will not automatically install the core dependencies. You will have to do that separately:
```
pip install numpy scipy
```
#### conda install
You can install the `galibrate` package from the `blakeaw` channel:
```
conda install -c blakeaw galibrate
```
NumPy and SciPy dependencies will be automatically installed with this version.

### Recommended additional software

The following software is not required for the basic operation of **GAlibrate**, but provides extra capabilities and features when installed.

#### Cython
**GAlibrate** includes an implementation of the core genetic algorithm that is written in   [Cython](https://cython.org/), which takes advantage of Cython-based optimizations and compilation to accelerate the algorithm. This version of genetic algorithm is used if Cython is installed.

#### Numba
**GAlibrate** also includes an implementation of the core genetic algorithm that takes advantage of [Numba](https://numba.pydata.org/)-based JIT compilation and optimization to accelerate the algorithm. This version of genetic algorithm is used if Numba is installed.

#### PySB
[PySB](http://pysb.org/) is needed to run PySB models, and it is therfore needed if you want to use tools from the galibrate.pysb_utils package.

------

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

------

# Documentation and Usage

### Quick Overview
Principally, **GAlibrate** defines the **GAO** (continuous **G**enetic **A**lgorithm-based **O**ptimizer ) class,
```python
from galibrate import GAO
```
which defines an object that can be used setup and run a continuous genetic algorithm-based optimization (i.e., a maximization) of a user-defined fitness function over the search space of a given set of (model) parameters.

Additionally, **GAlibrate** has a `pysb_utils` sub-package that provides the
`galibrate_it` module, which defines the GaoIt and GAlibrateIt classes (importable from the pysb_utils package level),
```python
from galibrate.pysb_utils import GaoIt, GAlibrateIt
```  
which create objects that abstract away some of the effort to setup and generate GAO instances for PySB models; [examples/pysb_dimerization_model](./examples/pysb_dimerization_model) provides some
examples for using GaoIt and GAlibrateIt objects. The `galibrate_it` module can also be called from the command line to generate a template run script for a PySB model,
```python
python -m galibrate.pysb_utils.galibrate_it pysb_model.py output_path
```
which users can then modify to fit their needs.

### Examples
Additional example scripts that show how to setup and launch Genetic Algorithm runs using **GAlibrate** can be found under [examples](./examples).

------

# Contact

To report problems or bugs please open a
[GitHub Issue](https://github.com/blakeaw/GAlibrate/issues). Additionally, any
comments, suggestions, or feature requests for **GAlibrate** can also be submitted as a
[GitHub Issue](https://github.com/blakeaw/GAlibrate/issues).

------

# Citing

If you use the **GAlibrate** software in your research, please cite it. You can export the  **GAlibrate** citation in your preferred format from its [Zenodo DOI](https://doi.org/10.5281/zenodo.3345232) entry.

Also, please cite the following references as appropriate for software used with/via **GAlibrate**:

#### Packages from the SciPy ecosystem

These include NumPy and SciPy for which references can be obtained from:
https://www.scipy.org/citing.html

#### PySB
  1. Lopez, C. F., Muhlich, J. L., Bachman, J. A. & Sorger, P. K. Programming biological models in Python using PySB. Mol Syst Biol 9, (2013). doi:[10.1038/msb.2013.1](dx.doi.org/10.1038/msb.2013.1)
