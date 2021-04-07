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

import numpy as np
from piquasso._math.torontonian import torontonian


def test_torontonian_on_2_by_2_real_matrix():
    matrix = np.array(
        [
            [1, 500],
            [-500, 6],
        ],
        dtype=float,
    )

    assert np.isclose(torontonian(matrix), -0.998)


def test_torontonian_on_2_by_2_complex_matrix():
    matrix = np.array(
        [
            [1, 500j],
            [500j, 6],
        ],
        dtype=complex,
    )

    assert np.isclose(torontonian(matrix), -0.998)


def test_torontonian_on_4_by_4_real_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4],
            [2, 6, 7, 8],
            [-3, 7, 3, 4],
            [4, -8, -4, 8],
        ],
        dtype=complex,
    )

    assert np.isclose(torontonian(matrix), 0.6095591888010201)


def test_torontonian_on_4_by_4_complex_matrix():
    matrix = np.array(
        [
            [1j, 2, 3j, 4],
            [2, 6, 7, 8j],
            [3j, 7, 3, 4],
            [4, 8j, 4, 8j],
        ],
        dtype=complex,
    )

    assert np.isclose(torontonian(matrix), 0.5167533900792849)


def test_torontonian_on_6_by_6_complex_matrix():
    matrix = np.array(
        [
            [1, 2, 3, 4j, 5, 6j],
            [2, 6, 7j, 8, 9, 5],
            [3, 7j, 3, 4, 3, 7],
            [4j, 8, 4, 8, 2, 1],
            [5, 9, 3, 2, 2j, 0],
            [6j, 5, 7, 1, 0, 1],
        ],
        dtype=complex,
    )

    assert np.isclose(torontonian(matrix), -0.9387510649770083-0.25289835806458744j)
