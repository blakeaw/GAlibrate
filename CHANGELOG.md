# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] - yyyy-mm-dd

N/A

### Added

### Changed

### Fixed

## [0.6.0] - 2020-06-21

### Added
- core GA now returns an array with fitness value of the fittest individual from each generation which can be accessed from the GAO property `GAO.best_fitness_per_generation`.

### Fixed
- Bug fix in core GA for sorting the population before selection and mating.

## [0.5.0] - 2020-06-20

### Added
- Optional progress bar to monitor passage of generations during GAO run that is only displayed if [tqdm](https://github.com/tqdm/tqdm) is installed.
- Optional [multiprocessing](https://docs.python.org/2/library/multiprocessing.html) based parallelism when evaluating the fitness function over the population during a GAO run.
 
