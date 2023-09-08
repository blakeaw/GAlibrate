# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] - yyyy-mm-dd

N/A

### Added

### Changed

### Fixed

## [0.7.0] - 2023-04-27 to 2023-08-31

### Added
 - New `benchmarks` module that defines functions often used to test and benchmark single-objective optimization applications.
 - New examples for each of the functions defined in the `benchmarks` functions.
 - 3-point line example.
 - PySB double-enzymatic model example.
 - Functions to resume/continue GAO runs for additional generations: `GAO.resume` and underlying functions `run_gao_py.continue_gao`, `run_gao_numba.continue_gao`, and `run_gao_cython.continue_gao`. 
 - `run_gao_julia` which ports key functions to Julia via PyJulia as another alternative to Numba or Cython acceleration. 
 - Notebook `01_scaling-performance` in a new `notebooks` directory. 
 - `tests` directory.
 - Test code for the `galibrate.benchmarks` and `galibrate.sampled_parameters`.
 - Test code for `galibrate.gao` and the `galibrate.gao.GAO` class, as well as different integrations using the Python, Cython, Numba, and Julia backend versions: `test_gao`, `test_gao_py`, `test_gao_numba`, `test_gao_cython`, and `test_gao_julia`.
 - Test code for the `GaoIt` class in the pysb submodule `galibrate.pysb.galibrate_it`: test code in `test_pysb`
 - Test code for the `galibrate.run_gao_numba`, `galibrate.run_gao_julia`, and `galibrate.run_gao_cython` modules: test code in `test_rungaonumba`, `test_rungaojulia`, and `test_rungaocython`, respectively.
 - New functions in `galibrate.gao` that load specific backend version: `_set_run_gao_numba`, etc. 

### Changed
  - The setup.py uses setuptools now instead of distutils. The new setup includes the Cython `.pyx` and Julia `.jl` files as data files in the package.  
  - Renamed the `galibrate.pysb_utils` to `galibrate.pysb`.
  - Formatted code using the Black format.

### Fixed
 - Corrected instances of `self.parm` in the `GaoIt` class to `self.parms`; fix for Issue https://github.com/blakeaw/GAlibrate/issues/8 
 - Added an `__init__.py` under `pysb_utils` that imports the `GaoIt` and `GAlibrateIt` classes; fix for Issue https://github.com/blakeaw/GAlibrate/issues/7
 - Error in the `GaoIt.mask` function which called `self.names` instead of correct fucntion call `self.names()`.
 - Error in the `GaoIt.add_all_nonkinetic_params` with misspelled `pysb_model.paramters` - correct: `pysb_model.parameters`
 - Switched instances of `np.int` to `np.int64` or `np.int_` (Cython module) for the following NumPy deprecation warning: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. 


## [0.6.0] - 2020-06-21

### Added
- core GA now returns an array with fitness value of the fittest individual from each generation which can be accessed from the GAO property `GAO.best_fitness_per_generation`.

### Fixed
- Bug fix in core GA for sorting the population before selection and mating.

## [0.5.0] - 2020-06-20

### Added
- Optional progress bar to monitor passage of generations during GAO run that is only displayed if [tqdm](https://github.com/tqdm/tqdm) is installed.
- Optional [multiprocessing](https://docs.python.org/2/library/multiprocessing.html) based parallelism when evaluating the fitness function over the population during a GAO run.
 
