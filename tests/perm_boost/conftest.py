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
#
# Portions of this file are based on work by Bence Soóki-Tóth, used with
# permission and originally made available under the MIT License.
#
# Bence Soóki-Tóth. "Efficient calculation of permanent function gradients
# in photonic quantum computing simulations", Eötvös Loránd University, 2025.

import pytest

# perm_boost requires complex128 / uint64 inputs. Enable x64 here for the test
# session (the library itself no longer flips this global at import time).
try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass


# Run after the parent conftest hook so our jax_platforms override sticks.
@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if config.getoption("--platform") != "gpu":
        return

    try:
        import jax_plugins.xla_cuda12  # noqa: F401
    except ImportError:
        pytest.exit(
            "--platform=gpu was specified but the jax-cuda12 plugin is not "
            "installed in this environment. Install it with: "
            "pip install 'piquasso[jax-cuda12]'",
            returncode=2,
        )

    # Pin to "cuda" specifically; the parent sets "gpu" which also probes ROCm.
    import jax

    jax.config.update("jax_platforms", "cuda")

    # Probe the backend so init failures surface here, not mid-test.
    try:
        devices = jax.devices("cuda")
    except Exception as exc:
        pytest.exit(
            f"--platform=gpu was specified but the CUDA backend failed to "
            f"initialize: {type(exc).__name__}: {exc}\n"
            f"Check that an NVIDIA driver compatible with CUDA 12 is loaded "
            f"and that jax / jaxlib / jax-cuda12-plugin versions agree.",
            returncode=2,
        )
    if not devices:
        pytest.exit(
            "--platform=gpu was specified but JAX sees no CUDA devices. "
            "Confirm with `nvidia-smi` that a GPU is visible.",
            returncode=2,
        )


# Skip every perm_boost test if the underlying module cannot be imported.
# The import of ``piquasso.jax_extensions`` raises ImportError when the
# C++ extension has not been built.
try:
    import piquasso.jax_extensions  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
