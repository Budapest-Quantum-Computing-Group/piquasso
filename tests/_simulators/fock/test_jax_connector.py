#
# Copyright 2021-2024 Budapest Quantum Computing Group
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
        del sys.modules["jax"]


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
