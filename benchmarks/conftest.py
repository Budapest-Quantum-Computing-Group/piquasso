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

import piquasso as pq

import numpy as np
from scipy.stats import unitary_group


@pytest.fixture
def generate_unitary_matrix():
    def func(N):
        if N == 1:
            return np.array([[np.exp(np.random.rand() * 2 * np.pi * 1j)]])

        return np.array(unitary_group.rvs(N), dtype=complex)

    return func


@pytest.fixture
def d():
    return 5


@pytest.fixture
def cutoff():
    return 4


@pytest.fixture
def measurement_cutoff():
    """
    NOTE: In SF the measurement cutoff is 5, and couldn't be changed
    """
    return 5


@pytest.fixture
def example_gaussian_pq_program(d):
    with pq.Program() as program:
        for i in range(d):
            pq.Q(i) | pq.Squeezing(r=0.1) | pq.Displacement(r=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

    return program


@pytest.fixture
def example_purefock_pq_program(d):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        for i in range(d):
            pq.Q(i) | pq.Squeezing(r=0.1) | pq.Displacement(r=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

    return program


@pytest.fixture
def pq_config(cutoff, measurement_cutoff):
    return pq.Config(cutoff=cutoff, measurement_cutoff=measurement_cutoff)


@pytest.fixture
def pq_gaussian_simulator(d, pq_config):
    return pq.GaussianSimulator(d=d, config=pq_config)


@pytest.fixture
def example_pq_gaussian_state(pq_gaussian_simulator, example_gaussian_pq_program):
    state = pq_gaussian_simulator.execute(example_gaussian_pq_program).state

    return state


@pytest.fixture
def pq_purefock_simulator(d, pq_config):
    return pq.PureFockSimulator(d=d, config=pq_config)


@pytest.fixture
def example_pq_purefock_state(pq_purefock_simulator, example_purefock_pq_program):
    state = pq_purefock_simulator.execute(example_purefock_pq_program).state

    return state


@pytest.fixture
def sf():
    try:
        import strawberryfields
    except ImportError as error:
        known_error_message = "cannot import name 'simps' from 'scipy.integrate'"

        if known_error_message in error.msg:
            raise ImportError(
                "Install a previous version of 'scipy', since 'strawberryfields' is "
                "incompatible with newer versions of scipy."
            )
        else:
            raise error

    return strawberryfields


@pytest.fixture
def example_gaussian_sf_program_and_engine(d, sf):
    """
    NOTE: the covariance matrix SF is returning is half of ours...
    It seems that our implementation is OK, however.
    """

    program = sf.Program(d)
    engine = sf.Engine(backend="gaussian")

    with program.context as q:
        sf.ops.Sgate(0.1) | q[0]
        sf.ops.Sgate(0.1) | q[1]
        sf.ops.Sgate(0.1) | q[2]
        sf.ops.Sgate(0.1) | q[3]
        sf.ops.Sgate(0.1) | q[4]

        sf.ops.Dgate(1) | q[0]
        sf.ops.Dgate(1) | q[1]
        sf.ops.Dgate(1) | q[2]
        sf.ops.Dgate(1) | q[3]
        sf.ops.Dgate(1) | q[4]

        sf.ops.BSgate(0.0959408065906761, 0.06786053071484363) | (q[0], q[1])
        sf.ops.BSgate(0.7730047654405018, 1.453770233324797) | (q[2], q[3])
        sf.ops.BSgate(1.0152680371119776, 1.2863559998816205) | (q[1], q[2])
        sf.ops.BSgate(1.3205517879465705, 0.5236836466492961) | (q[3], q[4])
        sf.ops.BSgate(4.394480318177715, 4.481575657714487) | (q[0], q[1])
        sf.ops.BSgate(2.2300919706807534, 1.5073556513699888) | (q[2], q[3])
        sf.ops.BSgate(2.2679037068773673, 1.9550229282085838) | (q[1], q[2])
        sf.ops.BSgate(3.340269832485504, 3.289367083610399) | (q[3], q[4])

    return program, engine


@pytest.fixture
def example_fock_sf_program_and_engine(d, cutoff, sf):
    """
    NOTE: the covariance matrix SF is returning is half of ours...
    It seems that our implementation is OK, however.
    """

    program = sf.Program(d)
    engine = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})

    with program.context as q:
        sf.ops.Sgate(0.1) | q[0]
        sf.ops.Sgate(0.1) | q[1]
        sf.ops.Sgate(0.1) | q[2]
        sf.ops.Sgate(0.1) | q[3]
        sf.ops.Sgate(0.1) | q[4]

        sf.ops.Dgate(1) | q[0]
        sf.ops.Dgate(1) | q[1]
        sf.ops.Dgate(1) | q[2]
        sf.ops.Dgate(1) | q[3]
        sf.ops.Dgate(1) | q[4]

        sf.ops.BSgate(0.0959408065906761, 0.06786053071484363) | (q[0], q[1])
        sf.ops.BSgate(0.7730047654405018, 1.453770233324797) | (q[2], q[3])
        sf.ops.BSgate(1.0152680371119776, 1.2863559998816205) | (q[1], q[2])
        sf.ops.BSgate(1.3205517879465705, 0.5236836466492961) | (q[3], q[4])
        sf.ops.BSgate(4.394480318177715, 4.481575657714487) | (q[0], q[1])
        sf.ops.BSgate(2.2300919706807534, 1.5073556513699888) | (q[2], q[3])
        sf.ops.BSgate(2.2679037068773673, 1.9550229282085838) | (q[1], q[2])
        sf.ops.BSgate(3.340269832485504, 3.289367083610399) | (q[3], q[4])

    return program, engine


@pytest.fixture
def example_sf_gaussian_state(example_gaussian_sf_program_and_engine):
    program, engine = example_gaussian_sf_program_and_engine
    results = engine.run(program)

    return results.state


@pytest.fixture
def example_sf_fock_state(example_fock_sf_program_and_engine):
    program, engine = example_fock_sf_program_and_engine
    results = engine.run(program)

    return results.state
