Í# Development Guide

## Development requirements

First and foremost, a `python3.6` is needed to be installed on your machine.
For the case of having a system python with version >=3.7, `pyenv` is recommended.

The `eigen3` C++ library is needed to be installed for the
`thewalrus` (dependency of `strawberryfields`).

The deb package is named `libeigen3-dev`, so issue
```
sudo apt-get install libeigen3-dev
```
on a machine running Ubuntu/Debian (the rpm package is named `eigen3-devel`).

Additionally, this project uses `tox` to manage virtualenvs, install it with
```
pip install tox
```

For packaging, `poetry` is used:
```
pip install poetry
```

## Testing

Run tests with
```
tox
```
