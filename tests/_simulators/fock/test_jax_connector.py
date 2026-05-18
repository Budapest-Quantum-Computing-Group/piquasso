#
# Copyright 2021-2026 Budapest Quantum Computing Group
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
import builtins

import numpy as np
import pytest


@pytest.fixture
def raise_ImportError_when_importing_jax():
    real_import = builtins.__import__

    def fake_import(name, globals, locals, fromlist, level):
        if name == "jax":
            raise ImportError()

        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import

    yield

    builtins.__import__ = real_import


@pytest.fixture
def unimport_jax():
    """
    Deletes `jax` from `sys.modules`.
    """

    if "jax" in sys.modules:
        jax = sys.modules["jax"]
        del sys.modules["jax"]

    yield

    sys.modules["jax"] = jax


def test_jaxConnector_imports_jax_if_installed():
    import piquasso as pq

    pq.JaxConnector()

    assert "jax" in sys.modules


def test_jaxConnector_raises_ImportError_if_jax_not_installed(
    raise_ImportError_when_importing_jax,
):
    import piquasso as pq

    with pytest.raises(ImportError) as error:
        pq.JaxConnector()

    assert error.value.args[0] == (
        "You have invoked a feature which requires 'jax'.\n"
        "You can install JAX via:\n"
        "\n"
        "pip install piquasso[jax]"
    )


def test_importing_Piquasso_does_not_import_jax(unimport_jax):
    import piquasso as pq  # noqa: F401

    assert "jax" not in sys.modules


def test_Piquasso_works_without_jax(
    unimport_jax,
    raise_ImportError_when_importing_jax,
):
    import piquasso as pq  # noqa: F401

    assert "jax" not in sys.modules


def test_jaxConnector_permanent_use_perm_boost_no_args():
    import piquasso as pq

    connector = pq.JaxConnector()

    with pytest.raises(TypeError, match="matrix argument is required"):
        connector.permanent(use_perm_boost=True)


def test_jaxConnector_permanent_use_perm_boost_missing_rows_cols(monkeypatch):
    """When the extension is importable, rows/cols must be supplied."""
    import piquasso as pq

    pytest.importorskip("piquasso.jax_extensions._perm_boost_core")

    connector = pq.JaxConnector()
    matrix = connector.np.eye(2, dtype=connector.np.complex128)

    with pytest.raises(TypeError, match="requires rows and cols"):
        connector.permanent(matrix, use_perm_boost=True)


def test_jaxConnector_permanent_use_perm_boost_happy_path():
    import piquasso as pq

    pytest.importorskip("piquasso.jax_extensions._perm_boost_core")

    connector = pq.JaxConnector()
    matrix = connector.np.array([[1.0, 2.0], [3.0, 4.0]], dtype=connector.np.complex128)
    rows = connector.np.ones(2, dtype=connector.np.uint64)
    cols = connector.np.ones(2, dtype=connector.np.uint64)

    result = connector.permanent(matrix, rows, cols, use_perm_boost=True)

    # permanent of [[1,2],[3,4]] = 1*4 + 2*3 = 10
    assert np.isclose(complex(result), 10.0)


def test_jaxConnector_permanent_use_perm_boost_falls_back_when_extension_absent(
    monkeypatch,
):
    """If the extension import fails, fall through to the pure-JAX permanent
    with a complex128 dtype promotion."""
    import piquasso as pq

    real_import = builtins.__import__

    def fake_import(name, globals, locals, fromlist, level):
        if name == "piquasso.jax_extensions.permanent":
            raise ImportError("simulated missing extension")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    connector = pq.JaxConnector()
    matrix = connector.np.array([[1.0, 2.0], [3.0, 4.0]], dtype=connector.np.float64)
    rows = connector.np.ones(2, dtype=connector.np.uint64)
    cols = connector.np.ones(2, dtype=connector.np.uint64)

    result = connector.permanent(matrix, rows, cols, use_perm_boost=True)

    assert np.isclose(complex(result), 10.0)
