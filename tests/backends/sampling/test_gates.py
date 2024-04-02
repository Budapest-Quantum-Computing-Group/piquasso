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

import numpy as np

import piquasso as pq


def test_program():
    U = np.array([[0.5, 0, 0], [0, 0.5j, 0], [0, 0, -1]], dtype=complex)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0, 1) | pq.Beamsplitter(0.5)
        pq.Q(1, 2, 3) | pq.Interferometer(U)
        pq.Q(3) | pq.Phaseshifter(0.5)
        pq.Q(4) | pq.Phaseshifter(0.5)
        pq.Q() | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5)
    result = simulator.execute(program, shots=10)

    assert len(result.samples) == 10


def test_interferometer():
    U = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=complex)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(4, 3, 1) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_interferometer = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 9, 0, 8, 7],
            [0, 0, 1, 0, 0],
            [0, 6, 0, 5, 4],
            [0, 3, 0, 2, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_phaseshifter():
    phi = np.pi / 2

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(2) | pq.Phaseshifter(phi)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    x = np.exp(1j * phi)
    expected_interferometer = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, x, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_beamsplitter():
    theta = np.pi / 4
    phi = np.pi / 3

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(1, 3) | pq.Beamsplitter(theta, phi)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    t = np.cos(theta)
    r = np.exp(1j * phi) * np.sin(theta)
    rc = np.conj(r)
    expected_interferometer = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, t, 0, -rc, 0],
            [0, 0, 1, 0, 0],
            [0, r, 0, t, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_lossy_program():
    r"""
    This test checks the average number of particles in the lossy BS.
    We expect average number to be smaller than initial one.
    """
    losses = 0.5

    d = 5
    simulator = pq.SamplingSimulator(d=d)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        for i in range(d):
            pq.Q(i) | pq.Loss(losses)

        pq.Q(0) | pq.Loss(transmissivity=0.0)
        pq.Q() | pq.ParticleNumberMeasurement()

    result = simulator.execute(program, shots=1)
    sample = result.samples[0]
    assert sum(sample) < sum(result.state.initial_state)


@pytest.mark.monkey
def test_LossyInterferometer_decreases_particle_number(generate_unitary_matrix):
    d = 5

    singular_values = np.array([0.0, 0.2, 0.3, 0.4, 0.5])

    lossy_interferometer_matrix = (
        generate_unitary_matrix(d)
        @ np.diag(singular_values)
        @ generate_unitary_matrix(d)
    )

    simulator = pq.SamplingSimulator(d=d)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q() | pq.LossyInterferometer(lossy_interferometer_matrix)
        pq.Q() | pq.ParticleNumberMeasurement()

    result = simulator.execute(program, shots=1)
    sample = result.samples[0]
    assert sum(sample) < sum(result.state.initial_state)


@pytest.mark.monkey
def test_LossyInterferometer_is_equivalent_to_Loss_and_Interferometers(
    generate_unitary_matrix,
):
    d = 5

    singular_values = np.array([0.0, 0.2, 0.3, 0.4, 0.5])

    first_unitary = generate_unitary_matrix(d)
    second_unitary = generate_unitary_matrix(d)

    lossy_interferometer_matrix = (
        first_unitary @ np.diag(singular_values) @ second_unitary
    )

    simulator = pq.SamplingSimulator(d=d)

    with pq.Program() as program_using_lossy_interferometer:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q() | pq.LossyInterferometer(lossy_interferometer_matrix)

    state_obtained_via_lossy_interferometer = simulator.execute(
        program_using_lossy_interferometer
    ).state

    with pq.Program() as program_using_loss:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q() | pq.Interferometer(second_unitary)

        for mode, loss in enumerate(singular_values):
            pq.Q(mode) | pq.Loss(loss)

        pq.Q() | pq.Interferometer(first_unitary)

    state_obtained_via_loss = simulator.execute(program_using_loss).state

    assert state_obtained_via_lossy_interferometer == state_obtained_via_loss


@pytest.mark.monkey
def test_LossyInterferometer_raises_InvalidParameter_for_invalid_matrix(
    generate_unitary_matrix,
):
    d = 5

    singular_values_out_of_bound = np.array([42, 0.2, 0.3, 0.4, 0.5])

    invalid_matrix = (
        generate_unitary_matrix(d)
        @ np.diag(singular_values_out_of_bound)
        @ generate_unitary_matrix(d)
    )

    with pytest.raises(pq.api.exceptions.InvalidParameter):
        pq.LossyInterferometer(invalid_matrix)


def test_Interferometer_fock_probabilities():
    U = np.array(
        [
            [
                -0.17959207 - 0.29175972j,
                -0.64550941 - 0.02897055j,
                -0.023922 - 0.38854671j,
                -0.07908932 + 0.35617293j,
                -0.39779191 + 0.14902272j,
            ],
            [
                -0.36896208 + 0.19468375j,
                -0.11545557 + 0.20434514j,
                0.25548079 + 0.05220164j,
                -0.51002161 + 0.38442256j,
                0.48106678 - 0.25210091j,
            ],
            [
                0.25912844 + 0.16131742j,
                0.11886251 + 0.12632645j,
                0.69028213 - 0.25734432j,
                0.01276639 + 0.05841739j,
                0.03713264 + 0.57364845j,
            ],
            [
                -0.20314019 - 0.18973473j,
                0.59146854 + 0.28605532j,
                -0.11096495 - 0.26870144j,
                -0.47290354 - 0.0489408j,
                -0.42459838 + 0.01554643j,
            ],
            [
                -0.5021973 - 0.53474291j,
                0.24524545 - 0.0741398j,
                0.37786104 + 0.10225255j,
                0.46696955 + 0.10636677j,
                0.07171789 - 0.09194236j,
            ],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 1, 0, 1])

        pq.Q(all) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    assert np.allclose(
        state.fock_probabilities,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00943787,
            0.00952411,
            0.00372613,
            0.01094254,
            0.02038039,
            0.00721118,
            0.01276527,
            0.00433598,
            0.00237178,
            0.00278069,
            0.00318524,
            0.00132885,
            0.00397591,
            0.00996051,
            0.01264881,
            0.00220331,
            0.0263324,
            0.00118848,
            0.00093221,
            0.00004332,
            0.00010173,
            0.03650928,
            0.00019833,
            0.00083033,
            0.00359651,
            0.01506536,
            0.00535646,
            0.00911342,
            0.00016192,
            0.00136495,
            0.01919529,
            0.00575667,
            0.00475163,
            0.00292093,
            0.01845835,
            0.00263738,
            0.01015263,
            0.00054558,
            0.01018948,
            0.00096209,
            0.00011264,
            0.00943751,
            0.00189028,
            0.00000646,
            0.02838532,
            0.01428406,
            0.00594266,
            0.0064234,
            0.00449348,
            0.00728765,
            0.00350418,
            0.00156008,
            0.00514618,
            0.00322227,
            0.0169176,
            0.01227155,
            0.00377727,
            0.03192492,
            0.00117325,
            0.00669423,
            0.00949246,
            0.00333097,
            0.00253143,
            0.00598864,
            0.00747331,
            0.0070525,
            0.01895052,
            0.00600548,
            0.00199403,
            0.01716476,
            0.00200791,
            0.00334997,
            0.00360096,
            0.00415943,
            0.00176133,
            0.00270693,
            0.00259121,
            0.00057382,
            0.02113925,
            0.00132904,
            0.00270719,
            0.00567207,
            0.0001369,
            0.00668861,
            0.00735136,
            0.00048563,
            0.00270623,
            0.00486821,
            0.03074534,
            0.0014593,
            0.00561172,
            0.00473769,
            0.00560528,
            0.00067681,
            0.01497427,
            0.00084121,
            0.00354908,
            0.02619859,
            0.00973237,
            0.00476371,
            0.00088827,
            0.00295503,
            0.01322995,
            0.01936212,
            0.01078059,
            0.00281038,
            0.00269666,
            0.01091767,
            0.00893086,
            0.03206584,
            0.01258918,
            0.00497573,
            0.01822315,
            0.03937057,
            0.00976562,
            0.00331421,
            0.00606718,
            0.01002473,
            0.01228917,
            0.00763452,
            0.00235901,
            0.01066404,
            0.01109757,
            0.02292789,
            0.00136971,
            0.0023765,
        ],
    )


def test_LossyInterferometer_fock_probabilities():
    U = np.array(
        [
            [
                -0.17959207 - 0.29175972j,
                -0.64550941 - 0.02897055j,
                -0.023922 - 0.38854671j,
                -0.07908932 + 0.35617293j,
                -0.39779191 + 0.14902272j,
            ],
            [
                -0.36896208 + 0.19468375j,
                -0.11545557 + 0.20434514j,
                0.25548079 + 0.05220164j,
                -0.51002161 + 0.38442256j,
                0.48106678 - 0.25210091j,
            ],
            [
                0.25912844 + 0.16131742j,
                0.11886251 + 0.12632645j,
                0.69028213 - 0.25734432j,
                0.01276639 + 0.05841739j,
                0.03713264 + 0.57364845j,
            ],
            [
                -0.20314019 - 0.18973473j,
                0.59146854 + 0.28605532j,
                -0.11096495 - 0.26870144j,
                -0.47290354 - 0.0489408j,
                -0.42459838 + 0.01554643j,
            ],
            [
                -0.5021973 - 0.53474291j,
                0.24524545 - 0.0741398j,
                0.37786104 + 0.10225255j,
                0.46696955 + 0.10636677j,
                0.07171789 - 0.09194236j,
            ],
        ]
    )

    singular_values = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    lossy_interferometer_matrix = U @ np.diag(singular_values) @ U @ U.T

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 1, 0, 1])

        pq.Q(all) | pq.LossyInterferometer(lossy_interferometer_matrix)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    assert np.allclose(
        state.fock_probabilities,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00000817,
            0.00001853,
            0.00011531,
            0.00000414,
            0.00015421,
            0.00000895,
            0.00015489,
            0.00000074,
            0.00028061,
            0.00018395,
            0.00009286,
            0.0005873,
            0.00000329,
            0.00010945,
            0.00055886,
            0.00000038,
            0.00001363,
            0.00000067,
            0.00007211,
            0.00004254,
            0.0000026,
            0.00061655,
            0.00000101,
            0.00000061,
            0.00110988,
            0.00015013,
            0.00029921,
            0.00044755,
            0.00007452,
            0.00085563,
            0.00001532,
            0.0000001,
            0.00012006,
            0.00058028,
            0.00054547,
            0.0000063,
            0.00005692,
            0.00000133,
            0.00001646,
            0.00006361,
            0.0000311,
            0.00016344,
            0.00000034,
            0.00001476,
            0.00053404,
            0.0004333,
            0.00024504,
            0.00029713,
            0.00000085,
            0.00018765,
            0.00100461,
            0.0000003,
            0.00000541,
            0.00004598,
            0.00237147,
            0.00037026,
            0.00031803,
            0.00000723,
            0.00023062,
            0.0000086,
            0.00034283,
            0.00000409,
            0.00121991,
            0.00036429,
            0.00023933,
            0.00000002,
            0.00000252,
            0.0012302,
            0.00053479,
            0.00220692,
            0.00000628,
            0.00014024,
            0.00000078,
            0.00009678,
            0.00059082,
            0.00001539,
            0.00050913,
            0.00000096,
            0.00001018,
            0.00014861,
            0.00061172,
            0.00006556,
            0.00010699,
            0.00004406,
            0.00005206,
            0.00150149,
            0.00000042,
            0.00002822,
            0.00003041,
            0.0019066,
            0.00044064,
            0.00008295,
            0.00010511,
            0.00045442,
            0.00003943,
            0.00103867,
            0.00001407,
            0.00038873,
            0.00031623,
            0.0003676,
            0.00000003,
            0.00001066,
            0.00004472,
            0.00016568,
            0.00145667,
            0.00016119,
            0.00005173,
            0.00007859,
            0.00045526,
            0.00017394,
            0.00016566,
            0.00010343,
            0.00002395,
            0.00013923,
            0.0001394,
            0.00000056,
            0.00013875,
            0.00231979,
            0.00102331,
            0.0003841,
            0.0,
            0.00000047,
            0.00004096,
            0.00296666,
            0.00278055,
            0.00000124,
        ],
    )
