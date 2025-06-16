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

from unittest import mock


def test_import_piquasso_works_without_tensorflow_dependency():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        import piquasso

        help(piquasso)


def test_import_piquasso_works_without_jax_dependency():
    with mock.patch.dict(sys.modules, {"jax": None}):
        import piquasso

        help(piquasso)


def test_import_piquasso_works_without_matplotlib_dependency():
    with mock.patch.dict(sys.modules, {"matplotlib": None}):
        import piquasso

        help(piquasso)
