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

import numpy as np

import piquasso as pq


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_measure_particle_number_on_one_mode(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 2)) * 3j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(1, 0, 1)) * (- 3j)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 1j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (- 1j)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)) * 2j
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)) * (- 2j)

        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) * 2 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 1 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) * 3 / 6

        pq.Q(2) | pq.ParticleNumberMeasurement()

    result = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (1, ) or sample == (2, )

    if sample == (1, ):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                1/3 * pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
                4j * pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)),
                -2j * pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)),
                -4j * pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)),
                2 / 3 * pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)),
                2j * pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)),
            ]
        )

    elif sample == (2, ):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2))
            ]
        )

    assert program.state == expected_state


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_measure_particle_number_on_two_modes(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 2)) * 3j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(1, 0, 1)) * (- 3j)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 1j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (- 1j)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)) * 2j
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)) * (- 2j)

        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) * 2 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 1 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) * 3 / 6

        pq.Q(1, 2) | pq.ParticleNumberMeasurement()

    result = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (0, 1) or sample == (1, 1) or sample == (0, 2)

    if sample == (0, 1):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
                pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (-6j),
                pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 6j,
            ]
        )

    elif sample == (1, 1):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)),
            ]
        )

    elif sample == (0, 2):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2))
            ]
        )

    assert program.state == expected_state


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_measure_particle_number_on_all_modes(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 0)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(0, 0, 1)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q() | pq.ParticleNumberMeasurement()

    result = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)

    sample = result.samples[0]
    assert sample == (0, 0, 0) or sample == (0, 0, 1) or sample == (1, 0, 0)

    if sample == (0, 0, 0):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)),
            ]
        )

    elif sample == (0, 0, 1):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
            ]
        )

    elif sample == (1, 0, 0):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)),
            ]
        )

    assert program.state == expected_state


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_measure_particle_number_with_multiple_shots(StateClass):
    shots = 4

    with pq.Program() as preparation:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 0)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(0, 0, 1)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q() | pq.ParticleNumberMeasurement()

    result = program.execute(shots=shots)

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(result.samples) == shots
