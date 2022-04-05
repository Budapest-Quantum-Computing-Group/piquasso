#
# Copyright 2021 Budapest Quantum Computing Group
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
    get_index_in_particle_subspace,
    get_index_in_fock_space,
)


def test_get_index_in_particle_subspace_for_0_particles():
    assert get_index_in_particle_subspace((0, 0, 0)) == 0


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 1), 0),
        ((0, 1, 0), 1),
        ((1, 0, 0), 2),
    ],
)
def test_get_index_in_particle_subspace_for_1_particle(vector, index):
    assert get_index_in_particle_subspace(vector) == index


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 2), 0),
        ((0, 1, 1), 1),
        ((0, 2, 0), 2),
        ((1, 0, 1), 3),
        ((1, 1, 0), 4),
        ((2, 0, 0), 5),
    ],
)
def test_get_index_in_particle_subspace_for_2_particles(vector, index):
    assert get_index_in_particle_subspace(vector) == index


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 3), 0),
        ((0, 1, 2), 1),
        ((0, 2, 1), 2),
        ((0, 3, 0), 3),
        ((1, 0, 2), 4),
        ((1, 1, 1), 5),
        ((1, 2, 0), 6),
        ((2, 0, 1), 7),
        ((2, 1, 0), 8),
        ((3, 0, 0), 9),
    ],
)
def test_get_index_in_particle_subspace_for_3_particles(vector, index):
    assert get_index_in_particle_subspace(vector) == index


def get_index_in_fock_space_for_0_particles():
    assert get_index_in_fock_space((0, 0, 0)) == 0


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 1), 1),
        ((0, 1, 0), 2),
        ((1, 0, 0), 3),
    ],
)
def get_index_in_fock_space_for_1_particle(vector, index):
    assert get_index_in_fock_space(vector) == index


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 2), 4),
        ((0, 1, 1), 5),
        ((0, 2, 0), 6),
        ((1, 0, 1), 7),
        ((1, 1, 0), 8),
        ((2, 0, 0), 9),
    ],
)
def test_get_index_in_fock_space_for_2_particles(vector, index):
    assert get_index_in_fock_space(vector) == index


@pytest.mark.parametrize(
    "vector, index",
    [
        ((0, 0, 3), 10),
        ((0, 1, 2), 11),
        ((0, 2, 1), 12),
        ((0, 3, 0), 13),
        ((1, 0, 2), 14),
        ((1, 1, 1), 15),
        ((1, 2, 0), 16),
        ((2, 0, 1), 17),
        ((2, 1, 0), 18),
        ((3, 0, 0), 19),
    ],
)
def test_get_index_in_fock_space_for_3_particles(vector, index):
    assert get_index_in_fock_space(vector) == index
