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
def test_5050_beamsplitter(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=3)
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1))

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 3)

    program.execute()

    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.5, 0.5, 0, 0, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_beamsplitter(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=3)
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1))

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.6545085, 0.3454915, 0, 0, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_beamsplitter_multiple_particles(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=2, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(2, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(2, 0)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(2, 0), bra=(0, 2)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.16362712, 0.08637288,
            0.24672554, 0.17929466, 0.32397979
        ],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_beamsplitter_leaves_vacuum_unchanged(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=2, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0), bra=(0, 0)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 2
        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 4

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0.25,
            0.32725425, 0.17274575,
            0.10709534, 0.11306356, 0.02984109
        ],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_multiple_beamsplitters(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1))

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 5)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 6, phi=1.5 * np.pi)

    program.execute()

    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.75, 0.25, 0,
            0, 0, 0, 0, 0, 0
        ],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_multiple_beamsplitters_with_multiple_particles(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 1, 1)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 2)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 5)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 6, phi=1.5 * np.pi)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.1875, 0.0625, 0,
            0.234375, 0.15625, 0.1875, 0.109375, 0.0625, 0
        ],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_phaseshift(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=2, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 2
        pq.Q() | pq.DensityMatrix(ket=(1, 1), bra=(1, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(1, 1)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(1, 1), bra=(0, 2)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0) | pq.Phaseshifter(phi=np.pi / 3)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.25, 0, 0.5, 0.25, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_fourier(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=2, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 2
        pq.Q() | pq.DensityMatrix(ket=(1, 1), bra=(1, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(1, 1)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(1, 1), bra=(0, 2)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0) | pq.Fourier()

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.25, 0, 0.5, 0.25, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_mach_zehnder(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=2, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 1), bra=(0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(0, 2)) / 2
        pq.Q() | pq.DensityMatrix(ket=(1, 1), bra=(1, 1)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 2), bra=(1, 1)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(1, 1), bra=(0, 2)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0, 1) | pq.MachZehnder(int_=np.pi/3, ext=np.pi/4)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.0625, 0.1875, 0.04845345, 0.09690689, 0.60463966],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_beamsplitters_and_phaseshifters_with_multiple_particles(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 1, 1)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 2)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0) | pq.Phaseshifter(phi=np.pi/3)
        pq.Q(1) | pq.Phaseshifter(phi=np.pi/3)

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 5)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 6, phi=1.5 * np.pi)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.1875, 0.0625, 0,
            0.43324878, 0.02366748, 0.1875, 0.04308374, 0.0625, 0
        ],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_interferometer(StateClass):

    with pq.Program() as preparation:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) / 4
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 1, 1)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 2)) * np.sqrt(1/8)

    T = np.array(
        [
            [0.5, 0.53033009 + 0.53033009j, 0.21650635 + 0.375j],
            [-0.61237244 + 0.61237244j,  0.4330127, 0.24148146 + 0.06470476j],
            [0, -0.48296291 + 0.12940952j, 0.8660254]
        ]
    )

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q(0, 1, 2) | pq.Interferometer(matrix=T)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.1875, 0.0625, 0,
            0.01443139, 0.10696977, 0.32090931, 0.0192306, 0.11538358, 0.17307537
        ],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_kerr(StateClass):
    xi = np.pi / 3

    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=4)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 3), bra=(0, 0, 3)) * 1
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 3), bra=(0, 1, 2)) * (-1j)
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 2), bra=(0, 0, 3)) * 1j

        pq.Q(2) | pq.Kerr(xi=xi)

    program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert np.isclose(program.state.norm, 1)

    assert len(nonzero_elements) == 3

    assert np.isclose(nonzero_elements[0][0], 1)
    assert nonzero_elements[0][1] == ((0, 0, 3), (0, 0, 3))

    assert np.isclose(nonzero_elements[1][0], - 1j * np.exp(- 1j * xi))
    assert nonzero_elements[1][1] == ((0, 0, 3), (0, 1, 2))

    assert np.isclose(nonzero_elements[2][0], 1j * np.exp(1j * xi))
    assert nonzero_elements[2][1] == ((0, 1, 2), (0, 0, 3))


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_cross_kerr(StateClass):
    xi = np.pi / 3

    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=4)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 3), bra=(0, 0, 3)) * 1
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 3), bra=(0, 1, 2)) * (-1j)
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 2), bra=(0, 0, 3)) * 1j

        pq.Q(1, 2) | pq.CrossKerr(xi=xi)

    program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert np.isclose(program.state.norm, 1)

    assert len(nonzero_elements) == 3

    assert np.isclose(nonzero_elements[0][0], 1)
    assert nonzero_elements[0][1] == ((0, 0, 3), (0, 0, 3))

    assert np.isclose(nonzero_elements[1][0], 1j * np.exp(1j * xi))
    assert nonzero_elements[1][1] == ((0, 0, 3), (0, 1, 2))

    assert np.isclose(nonzero_elements[2][0], - 1j * np.exp(- 1j * xi))
    assert nonzero_elements[2][1] == ((0, 1, 2), (0, 0, 3))
