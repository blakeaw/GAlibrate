# Before getting started

Thank you for contributing! Bug reports, feature requests, bug fixes or other contributions are all welcomed. 

## Coding conventions

GitHub is used to host code, track issues, and accept pull requests. This project uses [semantic versioning](https://semver.org/) and keeps a [CHANGELOG](./CHANGELOG.md). Google style docstrings are used, and [Black formatting](https://black.readthedocs.io/en/stable/) is applied to code. For new functions, it is requested that type annotations be included in the function definition.    


# How to contribute


## Issues

One great non-code way to contribute is to open an issue. 

### Types of Issues

Here are the different types of Issues you can contribute:

  * :bug: **Bug Reports**: Problem, errors, or other issues where the code is not working as expected. Please use the `bug` label. 
  * :bulb: **Feauture Request/Suggestion**: Request or suggest some new functionality or an update/change to existing functionality. Please use the `enhancement` label.

### Creating a new issue

You can open issues here: [https://github.com/blakeaw/GAlibrate/issues](https://github.com/blakeaw/GAlibrate/issues). 

However, before creating a new issue, please check the existing issues first to see if your issue or a similar one has already been raised. If it has, please add a comment to the existing issue rather than creating a new duplicate issue; e.g., with a bug report commenting with something like "I am also experiencing this problem" along with any additional context about your specific environment and package versions should suffice. 

For all Bug Report issues, please provide context, including the environment and relevant package versions, code snippets for the offending code/use when applicable, a description of the expected outcome, and the actual outcome with associated error messages when applicable. 

 ### Support

If you have support questions you can email them to [blakeaw1102@gmail.com](mailto:blakeaw1102@gmail.com)

## Pull Requests

If you want to contribute code that fixes bugs or adds new features you can fork the repository and open a pull request as described below. 

However, before doing so, please ask first. You can do so by commenting on the relevant Issue. This helps prevent duplicated or wasted efforts. 

PR contribution steps:

1. [Fork the repo](https://github.com/blakeaw/GAlibrate/fork)
2. Create a new branch, e.g.:
   * **Bug Fix:** `fix/issue-number`, e.g. `fix/11`
   * **New Feature** `feature/new-feature`, e.g. `feature/foo-bar`   
3. For feature additions, please include additional tests for the new feature.
4. Run all the tests using pytest: `python -m pytest`
5. Once your branch passes all the tests, commit your changes, e.g. `git commit -am 'Add the new-feautre feature.'`
7. Push the branch to your fork, e.g. `git push origin feature/new-feature`
8. Create a new [Pull request](https://github.com/blakeaw/GAlibrate/pulls). Reference any relevant Issues in the PR description.

Please note that any code contributions will be licensed according to this project's [LICENSE](./LICENSE).  

## Code of Conduct

All contributions will be considered based solely on their quality and fit with the overall direction of the project.   

All contributors are expected to be kind and respectful to one another. Behavior that is harmful to your fellow contributors is not acceptable. 