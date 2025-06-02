#
# Copyright 2021-2025 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import pytest

# Check Python version and TensorFlow availability
PYTHON_VERSION = sys.version_info
TENSORFLOW_AVAILABLE = False
JAX_AVAILABLE = False

try:
    import tensorflow as tf  # noqa: F401
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

try:
    import jax  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    pass


def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers",
        "tensorflow: mark test as requiring TensorFlow"
    )
    config.addinivalue_line(
        "markers",
        "jax: mark test as requiring JAX"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require TensorFlow or JAX if not available or incompatible."""
    for item in items:
        # Skip TensorFlow tests if not available or on Python 3.13+
        if "tensorflow" in item.keywords:
            if not TENSORFLOW_AVAILABLE:
                marker = pytest.mark.skip(reason="Test requires TensorFlow")
                item.add_marker(marker)
            elif PYTHON_VERSION >= (3, 13):
                marker = pytest.mark.skip(reason="TensorFlow not supported on Python 3.13+")
                item.add_marker(marker)
        # Skip JAX tests if not available
        elif "jax" in item.keywords and not JAX_AVAILABLE:
            marker = pytest.mark.skip(reason="Test requires JAX")
            item.add_marker(marker)


@pytest.fixture(scope="session")
def tf():
    """Fixture to provide TensorFlow or skip the test if not available."""
    if not TENSORFLOW_AVAILABLE:
        pytest.skip("TensorFlow is not installed")
    if PYTHON_VERSION >= (3, 13):
        pytest.skip("TensorFlow is not supported on Python 3.13+")
    
    import tensorflow as tf
    return tf


@pytest.fixture(scope="session")
def jax():
    """Fixture to provide JAX or skip the test if not available."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX is not installed")
    
    import jax
    return jax


import os
import sys
import pytest
import numpy as np

import piquasso as pq

from scipy.special import comb

from scipy.stats import unitary_group

from pathlib import Path

from scipy.linalg import polar, coshm, sinhm, logm

from piquasso._simulators.connectors import NumpyConnector



import sys
from pathlib import Path

# tests/conftest.py
import pytest
import sys


import pytest
import sys


def is_tensorflow_available():
    """Check if TensorFlow is importable and compatible with current Python version."""
    if sys.version_info >= (3, 13):
        return False, "TensorFlow not supported on Python 3.13+"
    
    try:
        import tensorflow as tf  # noqa: F401
        return True, None
    except ImportError:
        return False, "TensorFlow is not installed"
    except Exception as e:
        return False, f"Error importing TensorFlow: {str(e)}"


@pytest.fixture(scope="session")
def tf():
    """
    Fixture that provides a TensorFlow session for tests.
    Skips the test if TensorFlow is not available or not compatible.
    """
    is_available, reason = is_tensorflow_available()
    if not is_available:
        pytest.skip(reason)
    
    import tensorflow as tf
    
    # Configure TensorFlow to use CPU and disable GPU if needed
    tf.config.set_visible_devices([], 'GPU')
    
    return tf


def pytest_collection_modifyitems(config, items):
    """Handle TensorFlow test markers."""
    is_available, reason = is_tensorflow_available()
    
    skip_tf = pytest.mark.skip(reason=reason)
    for item in items:
        if "tensorflow" in item.keywords and not is_available:
            item.add_marker(skip_tf)


@pytest.fixture(autouse=True)
def skip_tensorflow_marked_tests(request):
    """Auto-skip tests marked with @pytest.mark.tensorflow on Python 3.13+"""
    if sys.version_info >= (3, 13) and "tensorflow" in request.keywords:
        pytest.skip("TensorFlow tests disabled on Python 3.13")
                
@pytest.fixture(autouse=True)
def _set_printoptions_for_debugging():
    np.set_printoptions(suppress=True, linewidth=200, precision=6)


@pytest.fixture
def generate_symmetric_matrix():
    def func(N):
        A = np.random.rand(N, N)

        return A + A.T

    return func


@pytest.fixture
def generate_complex_symmetric_matrix(generate_symmetric_matrix):
    def func(N):
        real = generate_symmetric_matrix(N)
        imaginary = generate_symmetric_matrix(N)

        return real + 1j * imaginary

    return func


@pytest.fixture
def generate_unitary_matrix():
    def func(N):
        if N == 1:
            return np.array([[np.exp(np.random.rand() * 2 * np.pi * 1j)]])

        return np.array(unitary_group.rvs(N), dtype=complex)

    return func


@pytest.fixture
def generate_random_positive_definite_matrix():
    def func(N):
        A = np.random.rand(N, N)
        return A @ A.transpose()

    return func


@pytest.fixture
def generate_hermitian_matrix(generate_unitary_matrix):
    def func(N):
        U = generate_unitary_matrix(N)

        return 1j * logm(U)

    return func


@pytest.fixture
def generate_skew_symmetric_matrix(generate_unitary_matrix):
    def func(N):
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)

        return A - A.T

    return func


@pytest.fixture
def generate_gaussian_transform(
    generate_complex_symmetric_matrix, generate_unitary_matrix
):
    def func(d):
        squeezing_matrix = generate_complex_symmetric_matrix(d)
        U, r = polar(squeezing_matrix)

        global_phase = generate_unitary_matrix(d)
        passive = global_phase @ coshm(r)
        active = global_phase @ sinhm(r) @ U.conj()

        return pq.GaussianTransform(passive=passive, active=active)

    return func


@pytest.fixture
def AssetHandler(request):
    class _AssetHandler:
        def __init__(self):
            testfile = request.module.__file__
            self._asset_dir = Path(testfile).parent / "assets"

            if not os.path.exists(self._asset_dir):
                os.makedirs(self._asset_dir)

            name = request.function.__name__ + "."

            if request.cls:
                name = request.cls.__name__ + "." + name

            self._asset_id = Path(os.path.splitext(testfile)[0]).name + "." + name

        def load(self, asset: str, loader: any = None):
            loader = loader or self._load_numpy_array

            filename = self._resolve_file_for_load(asset)

            return loader(filename)

        def save(self, asset: str, obj: any):
            filename = self._resolve_file_for_save(asset)

            self._write_repr(filename, obj)

        def _resolve_file_for_load(self, asset):
            pattern = self._asset_id + asset

            files = list(Path(self._asset_dir).glob(pattern))

            assert len(files) == 1, (
                f"There should be at least one asset file named '{pattern}' "
                f"under '{self._asset_dir}'."
            )

            return files[0]

        @staticmethod
        def _load_numpy_array(filename: str):
            with open(filename, "r") as f:
                data = f.read()

            # NOTE: This is needed for the evaluation with `eval` below, where `array`'s
            # are called.
            from numpy import array  # noqa: F401

            return eval(data)

        def _resolve_file_for_save(self, asset):
            return self._asset_dir / (self._asset_id + asset)

        @staticmethod
        def _write_repr(filename, obj):
            with open(filename, "w") as outfile:
                outfile.write(repr(obj))

    return _AssetHandler


@pytest.fixture
def assets(AssetHandler):
    return AssetHandler()


@pytest.fixture
def gaussian_state_assets(AssetHandler):
    class GaussianStateAssetHandler(AssetHandler):
        def __init__(self):
            super().__init__()

            self._asset = "expected_state"

        def load(self):
            filename = self._resolve_file_for_load(asset=self._asset)

            return self._load_gaussian_state(filename)

        def save(self, obj: any):
            state_map = {
                "class": "GaussianState",
                "properties": {
                    "xpxp_mean_vector": obj.xpxp_mean_vector,
                    "xpxp_covariance_matrix": obj.xpxp_covariance_matrix,
                },
            }
            np.set_printoptions(precision=16)

            filename = self._resolve_file_for_save(self._asset)

            np.set_printoptions(precision=8)

            self._write_repr(filename, state_map)

        @staticmethod
        def _load_gaussian_state(filename):
            """
            Note:
                Do not use the `eval` approach ever in production code, it's insecure!
            """
            with open(filename, "r") as f:
                data = f.read()

            # NOTE: This is needed for the evaluation with `eval` below, where `array`'s
            # are called.
            from numpy import array  # noqa: F401

            properties = eval(data)["properties"]

            d = len(properties["xpxp_mean_vector"]) // 2

            state = pq.GaussianState(d=d, connector=NumpyConnector())

            state.xpxp_mean_vector = properties["xpxp_mean_vector"]
            state.xpxp_covariance_matrix = properties["xpxp_covariance_matrix"]

            return state

    return GaussianStateAssetHandler()


def _get_click_distribution(d, n):
    dist = np.empty(shape=(d,))
    for c in range(1, d + 1):
        dist[c - 1] = comb(d, c) * comb(n - 1, n - c) / comb(n + d - 1, n)

    return dist


@pytest.fixture(scope="session")
def generate_random_fock_state():
    def func(d, n):
        occupation_numbers = np.zeros(shape=(d,), dtype=int)

        modes = tuple(range(d))
        c = d
        k = n

        while k > 0:
            dist = _get_click_distribution(c, k)

            c = np.random.choice(tuple(range(1, c + 1)), p=dist)
            k -= c

            modes = np.random.choice(modes, size=c, replace=False)

            occupation_numbers[modes] += 1

        return occupation_numbers

    return func
