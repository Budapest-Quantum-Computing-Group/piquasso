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
import builtins

import pytest


@pytest.fixture
def raise_ImportError_when_importing_tensorflow():
    real_import = builtins.__import__

    def fake_import(name, globals, locals, fromlist, level):
        if name == "tensorflow":
            raise ImportError()

        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import

    yield

    builtins.__import__ = real_import


@pytest.fixture
def unimport_tensorflow():
    """
    Deletes `tensorflow` from `sys.modules`.
    """
    if "tensorflow" in sys.modules:
        del sys.modules["tensorflow"]


@pytest.mark.tensorflow
def test_TensorflowConnector_imports_tensorflow_if_installed(tf):
    """Test that TensorFlow connector properly imports TensorFlow."""
    import piquasso as pq
    
    # This will raise an ImportError if TensorFlow is not properly imported
    connector = pq.TensorflowConnector()
    assert connector is not None


def test_TensorflowConnector_raises_ImportError_if_TensorFlow_not_installed(
    raise_ImportError_when_importing_tensorflow,
):
    """Test that proper error is raised when TensorFlow is not installed."""
    import piquasso as pq

    with pytest.raises(ImportError):
        pq.TensorflowConnector()


def test_importing_Piquasso_does_not_import_tensorflow(unimport_tensorflow):
    """Test that importing Piquasso doesn't automatically import TensorFlow."""
    import piquasso as pq  # noqa: F401

    assert "tensorflow" not in sys.modules


def test_Piquasso_works_without_TensorFlow(
    unimport_tensorflow,
    raise_ImportError_when_importing_tensorflow,
):
    """Test that Piquasso can be used without TensorFlow installed."""
    import piquasso as pq  # noqa: F401
    
    # This test passes if we can import piquasso without TensorFlow being available
    assert True
