<p align="center">
    <img width="70%" height="auto" src="https://raw.githubusercontent.com/Budapest-Quantum-Computing-Group/piquasso/main/piquasso_logo.svg" alt="Piquasso logo"/>
</p>

<p align="center">
  <a
    href="https://app.codecov.io/gh/Budapest-Quantum-Computing-Group/piquasso"
    style="text-decoration: none;"
  >
    <img
      alt="Coverage"
      src="https://img.shields.io/codecov/c/github/Budapest-Quantum-Computing-Group/piquasso"
    />
  </a>
  <a
    href="https://github.com/Budapest-Quantum-Computing-Group/piquasso/blob/main/LICENSE.txt"
    style="text-decoration: none;"
  >
    <img
      alt="License"
      src="https://img.shields.io/github/license/Budapest-Quantum-Computing-Group/piquasso"
    />
  </a>
  <a
    href="https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
    style="text-decoration: none;"
  >
    <img
      alt="Issues"
      src="https://img.shields.io/github/issues/Budapest-Quantum-Computing-Group/piquasso"
    />
  </a>
  <a
    href="https://github.com/Budapest-Quantum-Computing-Group/piquasso/pulls"
    style="text-decoration: none;"
  >
    <img
      alt="Pull requests"
      src="https://img.shields.io/github/issues-pr/Budapest-Quantum-Computing-Group/piquasso"
    />
  </a>
  <a
    href="https://github.com/Budapest-Quantum-Computing-Group/piquasso/actions"
    style="text-decoration: none;"
  >
    <img
      alt="Tests"
      src="https://github.com/Budapest-Quantum-Computing-Group/piquasso/actions/workflows/tests.yml/badge.svg"
    >
  </a>
</p>

<p align="center">
Piquasso is an open-source Python library for simulating photonic quantum computers.
</p>

> This is a research project, bugs can be expected. If you encounter any, please report
> it in the [Issues page](https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues).

## Basic example

```python
import numpy as np
import piquasso as pq

with pq.Program() as program:
    pq.Q(0) | pq.Displacement(
        r=np.sqrt(2), phi=np.pi / 4
    )  # Displace the state on mode 0.
    pq.Q(0, 1) | pq.Beamsplitter(
        theta=0, phi=np.pi / 2
    )  # Use a beamsplitter gate on modes 0, 1.

    pq.Q(0) | pq.ParticleNumberMeasurement()  # Measurement on mode 0.

simulator = pq.GaussianSimulator(
    d=3, config=pq.Config(hbar=2)
)  # Prepare a Gaussian vacuum state

result = simulator.execute(program, shots=10)  # Apply the program with 10 shots.

print("Resulting state:", result.state)
print("Detected samples:", result.samples)
```

This code outputs:
```
Resulting state: <piquasso._simulators.gaussian.state.GaussianState object at 0x7f3ef3604ac0>
Detected samples: [(0,), (2,), (1,), (2,), (2,), (4,), (1,), (1,), (4,), (1,)]
```

For more details, please refer to [the Piquasso documentation](https://piquasso.readthedocs.io/).

## Install

Piquasso and its dependencies can be installed via pip:

```
pip install piquasso
```

If you have problems installing Piquasso as above, try installing from source with

```
pip install --no-binary=:all: piquasso
```

When installing from source does not work on your machine, please open an issue in the
[Issues page](https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues).

If you wish to, you can also install
[piquassoboost](https://github.com/Budapest-Quantum-Computing-Group/piquassoboost) for performance improvement.

If you are doing research using Piquasso, please cite us as follows:
```
@article{Kolarovszki_2025,
   title={Piquasso: A Photonic Quantum Computer Simulation Software Platform},
   volume={9},
   ISSN={2521-327X},
   url={http://dx.doi.org/10.22331/q-2025-04-15-1708},
   DOI={10.22331/q-2025-04-15-1708},
   journal={Quantum},
   publisher={Verein zur Forderung des Open Access Publizierens in den Quantenwissenschaften},
   author={
      Kolarovszki, Zoltán
      and Rybotycki, Tomasz
      and Rakyta, Péter
      and Kaposi, Ágoston
      and Poór, Boldizsár
      and Jóczik, Szabolcs
      and Nagy, Dániel T. R.
      and Varga, Henrik
      and El-Safty, Kareem H.
      and Morse, Gregory
      and Oszmaniec, Michał
      and Kozsik, Tamás
      and Zimborás, Zoltán
   },
   year={2025},
   month=apr,
   pages={1708}
}
```

## Documentation

The documentation is avaliable at [https://piquasso.readthedocs.io/](https://piquasso.readthedocs.io/).

## How to contribute?

We welcome people who want to make contributions to Piquasso, be it big or small! If you are considering larger contributions to the source code, please contact us first.

We welcome bug reports, suggestions, and any feedback about Piquasso. To share these, open an issue or fill out the [Piquasso User Survey](https://forms.gle/urLy5S3kYs143ags6).

## Development guide

To install development dependencies, use:
```
pip install -e ".[dev]"
```

For document generation one should use
```
pip install -e ".[docs]"
```
Additionally, `pandoc` needs to be installed. To building the documentation, execute
`make html` in the `docs` folder. After a successful build, the documentation is
available under `docs/_build`.

For running files under `benchmarks/` or `scripts/`, please issue
```
pip install -e ".[benchmark]"
```

For building Piquasso, one also needs to install build dependencies:
```
pip install 'pybind11[global]' scikit-build-core cmake
```

### Linux

To build Piquasso for local development on Linux, run
```
cmake -B build -DCMAKE_INSTALL_PREFIX=$(pwd)
cmake --build build
cmake --install build
```
Here, the `-DCMAKE_INSTALL_PREFIX=$(pwd)` flag is needed to copy shared libraries into
the source tree for development.

### Windows

1. Open a terminal (e.g., Command Prompt or PowerShell).
2. Navigate (`cd`) to the root directory of the Piquasso project.
3. Run the following command:
```
build.bat
```

The `build.bat` script automatically runs all the required commands for building the project.

Alternatively, you can run the commands manually for building the project:
```
cmake -B build -DCMAKE_INSTALL_PREFIX="%cd%" -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Debug
cmake --install build --config Debug
```

### Testing

All tests and additional checks can be run using `tox`. After installation, run the
following command:
```
tox -e py312
```

Alternatively, you can run only the tests using `pytest`. After installation, run the
following command:
```
pytest tests
```

Besides testing, we have several additional checks:

1. We use an automatic formatting tool called `black`. To run automatic formatting, simply
   execute
   ```
   black .
   ```
2. Moreover, we also use a linter called `flake8`, which can be executed by issuing
   ```
   flake8
   ```
3. Finally, we use a static type checker called `mypy` on the `piquasso` folder:
   ```
   mypy piquasso
   ```
