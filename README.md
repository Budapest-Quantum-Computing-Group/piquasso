# Piquasso

![Codecov](https://img.shields.io/codecov/c/github/Budapest-Quantum-Computing-Group/piquasso)
![GitHub](https://img.shields.io/github/license/Budapest-Quantum-Computing-Group/piquasso)
![GitHub](https://img.shields.io/github/issues/Budapest-Quantum-Computing-Group/piquasso)
![GitHub](https://img.shields.io/github/issues-pr/Budapest-Quantum-Computing-Group/piquasso)

## A Python library for designing and simulating photonic quantum computers

Piquasso is based on a fine-grained model for photonic quantum computations. Besides
general computational models, Piquasso allows the application of special cases, such as
computations on particle number conserving or a pure Fock backend. The explicit use of
those special cases results in certain benefits. First of all, in Piquasso one can avoid
decoherence if the computation is theoretically guaranteed to be pure. Moreover, the
execution time and memory requirements of the computations can be significantly reduced
for those special cases.

## Installation

Piquasso and its dependencies can be installed via pip:

```
pip install piquasso
```

If you wish to, you can also install
[piquassoboost]( https://github.com/Budapest-Quantum-Computing-Group/piquassoboost) for 
performance improvement.

## Documentation

http://piquasso.com/

## How to contribute?

We welcome people who want to make contributions to Piquasso, be it big or small! If you
are considering larger contributions to the source code, please contact us first.

We also appreciate bug reports, suggestions, or any kind of idea regarding Piquasso.

## Development guide

The `eigen3` C++ library needs to be installed for the
`thewalrus` (dependency of `strawberryfields`).

On Ubuntu/Debian you can install it with

```
sudo apt-get install libeigen3-dev
```

Now, to install development dependencies, use:

```
pip install -rrequirements.txt
```

## Testing

Tests and additional checks can be run using `tox`. After installation, run the
following command:

```
tox -e py39
```

Alternatively, you can run only the tests using `pytest`. After installation, run the
following command:

```
pytest tests
```

