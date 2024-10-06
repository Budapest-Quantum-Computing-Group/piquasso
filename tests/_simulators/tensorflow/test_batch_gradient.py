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

import tensorflow as tf

import numpy as np

import piquasso as pq


connectors = (
    pq.TensorflowConnector(),
    pq.TensorflowConnector(decorate_with=tf.function),
)


@pytest.mark.parametrize("connector", connectors)
def test_batch_Beamsplitter_mean_position(connector):
    theta = tf.Variable(np.pi / 3)

    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with tf.GradientTape() as tape:
        with pq.Program() as batch_program:
            pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

            pq.Q() | pq.Beamsplitter(theta=theta, phi=np.pi / 3)

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=5), connector=connector
        )

        batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    batch_gradient = tape.jacobian(batch_mean_positions, theta)

    with tf.GradientTape() as tape:
        with pq.Program() as first_program:
            pq.Q() | first_preparation

            pq.Q() | pq.Beamsplitter(theta=theta, phi=np.pi / 3)

        first_mean_position = simulator.execute(first_program).state.mean_position(0)

    first_gradient = tape.gradient(first_mean_position, theta)

    with tf.GradientTape() as tape:
        with pq.Program() as second_program:
            pq.Q() | second_preparation

            pq.Q() | pq.Beamsplitter(theta=theta, phi=np.pi / 3)

        second_mean_position = simulator.execute(second_program).state.mean_position(0)

    second_gradient = tape.gradient(second_mean_position, theta)

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)

    assert np.allclose(batch_gradient[0], first_gradient)
    assert np.allclose(batch_gradient[1], second_gradient)


@pytest.mark.parametrize("connector", connectors)
def test_batch_Squeezing_mean_position(connector):
    r = tf.Variable(0.1)

    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with tf.GradientTape() as tape:
        with pq.Program() as batch_program:
            pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

            pq.Q(0) | pq.Squeezing(r=r)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=5), connector=connector
        )

        batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    batch_gradient = tape.jacobian(batch_mean_positions, r)

    with tf.GradientTape() as tape:
        with pq.Program() as first_program:
            pq.Q() | first_preparation

            pq.Q(0) | pq.Squeezing(r=r)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        first_mean_position = simulator.execute(first_program).state.mean_position(0)

    first_gradient = tape.gradient(first_mean_position, r)

    with tf.GradientTape() as tape:
        with pq.Program() as second_program:
            pq.Q() | second_preparation

            pq.Q(0) | pq.Squeezing(r=r)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        second_mean_position = simulator.execute(second_program).state.mean_position(0)

    second_gradient = tape.gradient(second_mean_position, r)

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)

    assert np.allclose(batch_gradient[0], first_gradient)
    assert np.allclose(batch_gradient[1], second_gradient)


@pytest.mark.parametrize("connector", connectors)
def test_batch_Displacement_mean_position(connector):
    r = tf.Variable(0.1)

    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with tf.GradientTape() as tape:
        with pq.Program() as batch_program:
            pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

            pq.Q(0) | pq.Displacement(r=r)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=5), connector=connector
        )

        batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    batch_gradient = tape.jacobian(batch_mean_positions, r)

    with tf.GradientTape() as tape:
        with pq.Program() as first_program:
            pq.Q() | first_preparation

            pq.Q(0) | pq.Displacement(r=r)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        first_mean_position = simulator.execute(first_program).state.mean_position(0)

    first_gradient = tape.gradient(first_mean_position, r)

    with tf.GradientTape() as tape:
        with pq.Program() as second_program:
            pq.Q() | second_preparation

            pq.Q(0) | pq.Displacement(r=r)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        second_mean_position = simulator.execute(second_program).state.mean_position(0)

    second_gradient = tape.gradient(second_mean_position, r)

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)

    assert np.allclose(batch_gradient[0], first_gradient)
    assert np.allclose(batch_gradient[1], second_gradient)


@pytest.mark.parametrize("connector", connectors)
def test_batch_Kerr_mean_position(connector):
    xi = tf.Variable(0.1)

    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with tf.GradientTape() as tape:
        with pq.Program() as batch_program:
            pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

            pq.Q(0) | pq.Kerr(xi=xi)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=5), connector=connector
        )

        batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    batch_gradient = tape.jacobian(batch_mean_positions, xi)

    with tf.GradientTape() as tape:
        with pq.Program() as first_program:
            pq.Q() | first_preparation

            pq.Q(0) | pq.Kerr(xi=xi)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        first_mean_position = simulator.execute(first_program).state.mean_position(0)

    first_gradient = tape.gradient(first_mean_position, xi)

    with tf.GradientTape() as tape:
        with pq.Program() as second_program:
            pq.Q() | second_preparation

            pq.Q(0) | pq.Kerr(xi=xi)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        second_mean_position = simulator.execute(second_program).state.mean_position(0)

    second_gradient = tape.gradient(second_mean_position, xi)

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)

    assert np.allclose(batch_gradient[0], first_gradient)
    assert np.allclose(batch_gradient[1], second_gradient)


@pytest.mark.parametrize("connector", connectors)
def test_batch_Phaseshifter_mean_position(connector):
    phi = tf.Variable(np.pi / 5)

    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with tf.GradientTape() as tape:
        with pq.Program() as batch_program:
            pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

            pq.Q(0) | pq.Phaseshifter(phi=phi)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=5), connector=connector
        )

        batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    batch_gradient = tape.jacobian(batch_mean_positions, phi)

    with tf.GradientTape() as tape:
        with pq.Program() as first_program:
            pq.Q() | first_preparation

            pq.Q(0) | pq.Phaseshifter(phi=phi)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        first_mean_position = simulator.execute(first_program).state.mean_position(0)

    first_gradient = tape.gradient(first_mean_position, phi)

    with tf.GradientTape() as tape:
        with pq.Program() as second_program:
            pq.Q() | second_preparation

            pq.Q(0) | pq.Phaseshifter(phi=phi)
            pq.Q() | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 3)

        second_mean_position = simulator.execute(second_program).state.mean_position(0)

    second_gradient = tape.gradient(second_mean_position, phi)

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)

    assert np.allclose(batch_gradient[0], first_gradient)
    assert np.allclose(batch_gradient[1], second_gradient)


@pytest.mark.parametrize("connector", connectors)
def test_batch_complex_circuit_mean_position(connector):
    theta1 = tf.Variable(np.pi / 3)
    phi1 = tf.Variable(np.pi / 4)

    phi_phaseshift = tf.Variable(np.pi / 5)

    r1 = tf.Variable(0.1)
    r2 = tf.Variable(0.2)

    alpha1 = tf.Variable(1.0)
    alpha2 = tf.Variable(0.5)

    theta2 = tf.Variable(np.pi / 3)
    phi2 = tf.Variable(np.pi / 4)

    theta2 = tf.Variable(np.pi / 3)
    phi2 = tf.Variable(np.pi / 4)

    xi1 = tf.Variable(0.1)
    xi2 = tf.Variable(0.15)

    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with tf.GradientTape() as tape:
        with pq.Program() as batch_program:
            pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

            pq.Q() | pq.Beamsplitter(theta=theta1, phi=phi1)
            pq.Q(0) | pq.Phaseshifter(phi=phi_phaseshift)

            pq.Q(0) | pq.Squeezing(r=r1)
            pq.Q(1) | pq.Squeezing(r=r2)

            pq.Q(0) | pq.Displacement(r=alpha1)
            pq.Q(1) | pq.Displacement(r=alpha2)

            pq.Q() | pq.Beamsplitter(theta=theta2, phi=phi2)

            pq.Q(0) | pq.Kerr(xi=xi1)
            pq.Q(1) | pq.Kerr(xi=xi2)

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=5), connector=connector
        )

        batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    parameters = [theta1, phi1, phi_phaseshift, r1, r2, alpha1, alpha2, theta2, phi2]

    batch_gradient = np.array(tape.jacobian(batch_mean_positions, parameters))

    with tf.GradientTape() as tape:
        with pq.Program() as first_program:
            pq.Q() | first_preparation

            pq.Q() | pq.Beamsplitter(theta=theta1, phi=phi1)
            pq.Q(0) | pq.Phaseshifter(phi=phi_phaseshift)

            pq.Q(0) | pq.Squeezing(r=r1)
            pq.Q(1) | pq.Squeezing(r=r2)

            pq.Q(0) | pq.Displacement(r=alpha1)
            pq.Q(1) | pq.Displacement(r=alpha2)

            pq.Q() | pq.Beamsplitter(theta=theta2, phi=phi2)

            pq.Q(0) | pq.Kerr(xi=xi1)
            pq.Q(1) | pq.Kerr(xi=xi2)

        first_mean_position = simulator.execute(first_program).state.mean_position(0)

    first_gradient = np.array(tape.gradient(first_mean_position, parameters))

    with tf.GradientTape() as tape:
        with pq.Program() as second_program:
            pq.Q() | second_preparation

            pq.Q() | pq.Beamsplitter(theta=theta1, phi=phi1)
            pq.Q(0) | pq.Phaseshifter(phi=phi_phaseshift)

            pq.Q(0) | pq.Squeezing(r=r1)
            pq.Q(1) | pq.Squeezing(r=r2)

            pq.Q(0) | pq.Displacement(r=alpha1)
            pq.Q(1) | pq.Displacement(r=alpha2)

            pq.Q() | pq.Beamsplitter(theta=theta2, phi=phi2)

            pq.Q(0) | pq.Kerr(xi=xi1)
            pq.Q(1) | pq.Kerr(xi=xi2)

        second_mean_position = simulator.execute(second_program).state.mean_position(0)

    second_gradient = np.array(tape.gradient(second_mean_position, parameters))

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)

    assert np.allclose(batch_gradient[:, 0], first_gradient)
    assert np.allclose(batch_gradient[:, 1], second_gradient)


@pytest.mark.parametrize("connector", connectors)
def test_batch_complex_circuit_mean_position_with_batch_apply(connector):
    theta1 = tf.Variable(np.pi / 3)
    phi1 = tf.Variable(np.pi / 4)

    phi_phaseshift = tf.Variable(np.pi / 5)

    r1 = tf.Variable(0.1)
    r2 = tf.Variable(0.2)

    alpha1 = tf.Variable(1.0)
    alpha2 = tf.Variable(0.5)

    theta2 = tf.Variable(np.pi / 3)
    phi2 = tf.Variable(np.pi / 4)

    theta2 = tf.Variable(np.pi / 3)
    phi2 = tf.Variable(np.pi / 4)

    xi1 = tf.Variable(0.1)
    xi2 = tf.Variable(0.15)

    xi1_intermediate = tf.Variable(0.3)
    xi2_intermediate = tf.Variable(0.4)

    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with tf.GradientTape() as tape:
        with pq.Program() as first_intermediate:
            pq.Q(0) | pq.Kerr(xi1_intermediate)

        with pq.Program() as second_intermediate:
            pq.Q(1) | pq.Kerr(xi2_intermediate)

        with pq.Program() as batch_program:
            pq.Q() | pq.BatchPrepare([first_preparation, second_preparation])

            pq.Q() | pq.Beamsplitter(theta=theta1, phi=phi1)
            pq.Q(0) | pq.Phaseshifter(phi=phi_phaseshift)

            pq.Q(0) | pq.Squeezing(r=r1)
            pq.Q(1) | pq.Squeezing(r=r2)

            pq.Q() | pq.BatchApply([first_intermediate, second_intermediate])

            pq.Q(0) | pq.Displacement(r=alpha1)
            pq.Q(1) | pq.Displacement(r=alpha2)

            pq.Q() | pq.Beamsplitter(theta=theta2, phi=phi2)

            pq.Q(0) | pq.Kerr(xi=xi1)
            pq.Q(1) | pq.Kerr(xi=xi2)

        simulator = pq.PureFockSimulator(
            d=2, config=pq.Config(cutoff=5), connector=connector
        )

        batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    parameters = [theta1, phi1, phi_phaseshift, r1, r2, alpha1, alpha2, theta2, phi2]

    batch_gradient = np.array(
        tape.jacobian(
            batch_mean_positions, parameters + [xi1_intermediate, xi2_intermediate]
        )
    )

    with tf.GradientTape() as tape:
        with pq.Program() as first_program:
            pq.Q() | first_preparation

            pq.Q() | pq.Beamsplitter(theta=theta1, phi=phi1)
            pq.Q(0) | pq.Phaseshifter(phi=phi_phaseshift)

            pq.Q(0) | pq.Squeezing(r=r1)
            pq.Q(1) | pq.Squeezing(r=r2)

            pq.Q(0) | pq.Kerr(xi1_intermediate)

            pq.Q(0) | pq.Displacement(r=alpha1)
            pq.Q(1) | pq.Displacement(r=alpha2)

            pq.Q() | pq.Beamsplitter(theta=theta2, phi=phi2)

            pq.Q(0) | pq.Kerr(xi=xi1)
            pq.Q(1) | pq.Kerr(xi=xi2)

        first_mean_position = simulator.execute(first_program).state.mean_position(0)

    first_gradient = np.array(
        tape.gradient(first_mean_position, parameters + [xi1_intermediate])
    )

    with tf.GradientTape() as tape:
        with pq.Program() as second_program:
            pq.Q() | second_preparation

            pq.Q() | pq.Beamsplitter(theta=theta1, phi=phi1)
            pq.Q(0) | pq.Phaseshifter(phi=phi_phaseshift)

            pq.Q(0) | pq.Squeezing(r=r1)
            pq.Q(1) | pq.Squeezing(r=r2)

            pq.Q(1) | pq.Kerr(xi2_intermediate)

            pq.Q(0) | pq.Displacement(r=alpha1)
            pq.Q(1) | pq.Displacement(r=alpha2)

            pq.Q() | pq.Beamsplitter(theta=theta2, phi=phi2)

            pq.Q(0) | pq.Kerr(xi=xi1)
            pq.Q(1) | pq.Kerr(xi=xi2)

        second_mean_position = simulator.execute(second_program).state.mean_position(0)

    second_gradient = np.array(
        tape.gradient(second_mean_position, parameters + [xi2_intermediate])
    )

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)

    assert np.allclose(
        batch_gradient[: len(parameters), 0], first_gradient[: len(parameters)]
    )
    assert np.isclose(batch_gradient[-2, 0], first_gradient[-1])
    assert np.isclose(batch_gradient[-1, 0], 0.0)

    assert np.allclose(
        batch_gradient[: len(parameters), 1], second_gradient[: len(parameters)]
    )
    assert np.isclose(batch_gradient[-1, 1], second_gradient[-1])
    assert np.isclose(batch_gradient[-2, 1], 0.0)
