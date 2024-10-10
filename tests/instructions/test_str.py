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

import numpy as np

import piquasso as pq


def test_str_of_preparations():
    assert str(pq.Vacuum()) == "Vacuum(modes=())"
    assert (
        str(pq.StateVector([1, 2]))
        == "StateVector(occupation_numbers=(1, 2), coefficient=1.0, modes=())"
    )

    assert (
        str(pq.StateVector([1, 2]).on_modes(0, 1))
        == "StateVector(occupation_numbers=(1, 2), coefficient=1.0, modes=(0, 1))"
    )

    assert (
        str(pq.StateVector([1, 2]).on_modes(0, 1) * 0.2)
        == "StateVector(occupation_numbers=(1, 2), coefficient=0.2, modes=(0, 1))"
    )


def test_str_of_gates():
    assert (
        str(pq.Phaseshifter(phi=np.pi / 3))
        == "Phaseshifter(phi=1.0471975511965976, modes=())"
    )

    assert (
        str(pq.Phaseshifter(phi=np.pi / 3).on_modes(1))
        == "Phaseshifter(phi=1.0471975511965976, modes=(1,))"
    )

    assert str(pq.Squeezing(r=1.0)) == "Squeezing(r=1.0, phi=0.0, modes=())"
    assert (
        str(pq.Displacement(r=1.0, phi=0.5)) == "Displacement(r=1.0, phi=0.5, modes=())"
    )

    assert (
        str(pq.Beamsplitter(theta=0.2, phi=0.3))
        == "Beamsplitter(theta=0.2, phi=0.3, modes=())"
    )
