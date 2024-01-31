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

import pytest

from piquasso._math.indices import (
    get_index_in_fock_space,
    get_index_in_fock_subspace,
)


def get_index_in_fock_space_for_0_particles():
    assert get_index_in_fock_space((0, 0, 0)) == 0


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 1), 3),
        ((0, 1, 0), 2),
        ((1, 0, 0), 1),
    ],
)
def get_index_in_fock_space_for_1_particle(vector, index):
    assert get_index_in_fock_space(vector) == index


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 2), 9),
        ((0, 1, 1), 8),
        ((0, 2, 0), 7),
        ((1, 0, 1), 6),
        ((1, 1, 0), 5),
        ((2, 0, 0), 4),
    ],
)
def test_get_index_in_fock_space_for_2_particles(vector, index):
    assert get_index_in_fock_space(vector) == index


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 3), 19),
        ((0, 1, 2), 18),
        ((0, 2, 1), 17),
        ((0, 3, 0), 16),
        ((1, 0, 2), 15),
        ((1, 1, 1), 14),
        ((1, 2, 0), 13),
        ((2, 0, 1), 12),
        ((2, 1, 0), 11),
        ((3, 0, 0), 10),
    ],
)
def test_get_index_in_fock_space_for_3_particles(vector, index):
    assert get_index_in_fock_space(vector) == index


def test_get_index_in_fock_subspace_for_0_particles():
    assert get_index_in_fock_subspace((0, 0, 0)) == 0


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 1), 2),
        ((0, 1, 0), 1),
        ((1, 0, 0), 0),
    ],
)
def test_get_index_in_fock_subspace_for_1_particle(vector, index):
    assert get_index_in_fock_subspace(vector) == index


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 2), 5),
        ((0, 1, 1), 4),
        ((0, 2, 0), 3),
        ((1, 0, 1), 2),
        ((1, 1, 0), 1),
        ((2, 0, 0), 0),
    ],
)
def test_get_index_in_fock_subspace_for_2_particles(vector, index):
    assert get_index_in_fock_subspace(vector) == index


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 3), 9),
        ((0, 1, 2), 8),
        ((0, 2, 1), 7),
        ((0, 3, 0), 6),
        ((1, 0, 2), 5),
        ((1, 1, 1), 4),
        ((1, 2, 0), 3),
        ((2, 0, 1), 2),
        ((2, 1, 0), 1),
        ((3, 0, 0), 0),
    ],
)
def test_get_index_in_fock_subspace_for_3_particles(vector, index):
    assert get_index_in_fock_subspace(vector) == index
