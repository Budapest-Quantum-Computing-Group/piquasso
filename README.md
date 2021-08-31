# Development Guide

## How to contribute?

Open an *Issue* or a *Pull request*.

## Development requirements

The `eigen3` C++ library is needed to be installed for the
`thewalrus` (dependency of `strawberryfields`).

The Ubuntu/Debian you can install it with
```
sudo apt-get install libeigen3-dev
```

Now, to enter a virtual environment and install all development dependencies, use
```
virtualenv .venv
source .venv/bin/activate
pip install -e .
pip install -rrequirements.txt
```

## Testing

Testing can be done with `tox` as well. Install `tox` and run tests with the following commands:
```
pip install tox
tox -e py39
```
