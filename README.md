# Development Guide

## Development requirements

First and foremost, a `python3.6` is needed to be installed on your machine.
For the case of having a system python with version >=3.7, `pyenv` is recommended.

The `eigen3` C++ library is needed to be installed for the
`thewalrus` (dependency of `strawberryfields`).

The deb package is named `libeigen3-dev`, so issue
```
sudo apt-get install libeigen3-dev
```
on a machine running Fedora/CentOS (the rpm package is named `eigen3-devel`).

`python3-venv` is important for the python virtual environment, install it on linux with:
```
sudo apt install python3-venv
```

Additionally, this project uses `tox` to manage virtualenvs, install it with:
```
pip install tox
```

For packaging, `poetry` is used.
Install it with the below command:
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```
`poetry` can be installed with `pip` as well, but the recommended way is to install it
with the above command. More info [here](https://python-poetry.org/docs/#installation).

The last step is to run the below command to download all the python dependencies (more
info [here](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)):
```
poetry install
```

## Running a program with poetry

A python program can be run inside poetries virtual environment with the below command
```
poetry run python <python file>
```
Or the virtual environment can be activated explicitly by running:
```
source `poetry env info --path`/bin/activate
```

## Testing

Run tests with
```
tox
```

## How to contribute?

We plan out and track all Piquasso issues through *Issues*. Feel free
to browse around and pick a sympathetic one to work on.
