from setuptools import setup, find_packages

# with open("README.md", encoding="utf8") as fh:
#     long_description = fh.read()

setup(
    name="galibrate",
    version="0.7.0",
    python_requires=">=3.10",
    install_requires=["numpy>=1.23.5", "scipy>=1.10.1"],
    extras_require={
        "cython": "cython>=0.29.33",
        "pysb": "pysb>=1.15.0",
        "numba": "numba>=0.56.4",
        "pyjulia": "julia>=0.6.1",
    },
    author="Blake A. Wilson",
    author_email="blakeaw1102@gmail.com",
    description="Python toolkit for continuous Genetic Algorithm optimization.",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/blakeaw/GAlibrate",
    packages=find_packages(),
    # Capture and include the Cython and Julia modules as data files.
    package_data={"": ["*.pyx", "*.jl"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    keywords=[
        "continuous genetic algorithm",
        "model calibration",
        "parameter estimation",
    ],
)
