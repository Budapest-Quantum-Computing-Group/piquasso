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

import jax
import pytest

from piquasso._math import perm_boost as _perm_boost


def pytest_addoption(parser):
    parser.addoption(
        "--platform", action="store", default="cpu", help="Choose platform: cpu or gpu"
    )


def pytest_configure(config):
    platform = config.getoption("--platform")
    jax.config.update("jax_platform_name", platform)


@pytest.fixture(scope="session", autouse=True)
def register_ffi_targets():
    jax.config.update("jax_enable_x64", True)

    for name, target in _perm_boost.registrations().items():
        jax.ffi.register_ffi_target(name, target)
