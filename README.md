<p align="center">
    <img src="https://raw.githubusercontent.com/Budapest-Quantum-Computing-Group/piquasso/main/piquasso_logo.svg" alt="Piquasso logo"/>
</p>

![Codecov](https://img.shields.io/codecov/c/github/Budapest-Quantum-Computing-Group/piquasso)
![GitHub](https://img.shields.io/github/license/Budapest-Quantum-Computing-Group/piquasso)
![GitHub](https://img.shields.io/github/issues/Budapest-Quantum-Computing-Group/piquasso)
![GitHub](https://img.shields.io/github/issues-pr/Budapest-Quantum-Computing-Group/piquasso)

## A Python library for designing and simulating photonic quantum computers

Piquasso is a simulator for photonic quantum computations.

> This is a research project, bugs can be expected. If you encounter any, please report
> it in the [Issues page](https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues).

## Installation

Piquasso and its dependencies can be installed via pip:

```
pip install piquasso
```

If you wish to, you can also install
[piquassoboost](https://github.com/Budapest-Quantum-Computing-Group/piquassoboost) for
performance improvement.

## Documentation

The documentation is avaliable at [docs.piquasso.com](https://docs.piquasso.com/).

## How to contribute?

We welcome people who want to make contributions to Piquasso, be it big or small! If you
are considering larger contributions to the source code, please contact us first.

We also appreciate bug reports, suggestions, or any kind of idea regarding Piquasso.

## Development guide

To install development dependencies, use:
```
pip install -e ".[dev]"
```

For document generation one should use
```
pip install -e ".[doc]"
```
and for running files under `benchmarks/` or `scripts/`, please issue
```
pip install -e ".[benchmark]"
```

## Testing

Tests and additional checks can be run using `tox`. After installation, run the
following command:

```
tox -e py312
```

Alternatively, you can run only the tests using `pytest`. After installation, run the
following command:

```
pytest tests
```
