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

import pytest
import numpy as np
from scipy.stats import unitary_group

import piquasso as pq

# Import TensorFlow only if available
pytest.importorskip("tensorflow")
import tensorflow as tf  # noqa: E402

# Import JAX only if available
try:
    from jax import grad  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from piquasso.decompositions.clements import (
    clements,
    inverse_clements,
    get_weights_from_decomposition,
    get_decomposition_from_weights,
    get_weights_from_interferometer,
    get_interferometer_from_weights,
    instructions_from_decomposition,
    get_decomposition_from_instructions,
)


pytestmark = pytest.mark.monkey


@pytest.fixture
def dummy_unitary():
    def func(d):
        return np.array(unitary_group.rvs(d))

    return func


def test_clements_decomposition_using_piquasso_SamplingSimulator(dummy_unitary):
    d = 3
    U = dummy_unitary(d)

    decomposition = clements(U, connector=pq.NumpyConnector())

    with pq.Program() as program_with_interferometer:
        pq.Q() | pq.StateVector(tuple([1] * d))

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        pq.Q() | pq.StateVector(tuple([1] * d))

        for operation in decomposition.beamsplitters:
            pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=operation.params[1])
            pq.Q(*operation.modes) | pq.Beamsplitter(operation.params[0], 0.0)

        for operation in decomposition.phaseshifters:
            pq.Q(operation.mode) | pq.Phaseshifter(operation.phi)

    simulator = pq.SamplingSimulator(d=d)

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


def test_clements_decomposition_using_piquasso_PureFockSimulator(dummy_unitary):
    d = 4
    U = dummy_unitary(d)

    decomposition = clements(U, connector=pq.NumpyConnector())

    occupation_numbers = (1, 1, 0, 0)

    with pq.Program() as program_with_interferometer:
        pq.Q() | pq.StateVector(occupation_numbers)

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        pq.Q() | pq.StateVector(occupation_numbers)

        for operation in decomposition.beamsplitters:
            pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=operation.params[1])
            pq.Q(*operation.modes) | pq.Beamsplitter(operation.params[0], 0.0)

        for operation in decomposition.phaseshifters:
            pq.Q(operation.mode) | pq.Phaseshifter(operation.phi)

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=sum(occupation_numbers) + 1)
    )

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


def test_clements_decomposition_using_piquasso_FockSimulator(dummy_unitary):
    d = 4
    U = dummy_unitary(d)

    decomposition = clements(U, connector=pq.NumpyConnector())

    with pq.Program() as program_with_interferometer:
        pq.Q() | pq.DensityMatrix((1, 0, 1, 0), (1, 0, 1, 0))

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        pq.Q() | pq.DensityMatrix((1, 0, 1, 0), (1, 0, 1, 0))

        for operation in decomposition.beamsplitters:
            pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=operation.params[1])
            pq.Q(*operation.modes) | pq.Beamsplitter(operation.params[0], 0.0)

        for operation in decomposition.phaseshifters:
            pq.Q(operation.mode) | pq.Phaseshifter(operation.phi)

    simulator = pq.FockSimulator(d=d, config=pq.Config(cutoff=3))

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


def test_clements_decomposition_using_piquasso_GaussianSimulator(dummy_unitary):
    d = 3
    U = dummy_unitary(d)

    decomposition = clements(U, connector=pq.NumpyConnector())

    squeezings = [0.1, 0.2, 0.3]

    with pq.Program() as program_with_interferometer:
        for mode, r in enumerate(squeezings):
            pq.Q(mode) | pq.Squeezing(r)

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        for mode, r in enumerate(squeezings):
            pq.Q(mode) | pq.Squeezing(r)

        for operation in decomposition.beamsplitters:
            pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=operation.params[1])
            pq.Q(*operation.modes) | pq.Beamsplitter(operation.params[0], 0.0)

        for operation in decomposition.phaseshifters:
            pq.Q(operation.mode) | pq.Phaseshifter(operation.phi)

    simulator = pq.GaussianSimulator(d=d)

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


def test_instructions_from_decomposition_using_piquasso_GaussianSimulator(
    dummy_unitary,
):
    d = 3
    U = dummy_unitary(d)

    decomposition = clements(U, connector=pq.NumpyConnector())

    squeezings = [0.1, 0.2, 0.3]

    with pq.Program() as program_with_interferometer:
        for mode, r in enumerate(squeezings):
            pq.Q(mode) | pq.Squeezing(r)

        pq.Q() | pq.Interferometer(matrix=U)

    program_with_decomposition = pq.Program(
        instructions=[
            pq.Squeezing(r).on_modes(mode) for mode, r in enumerate(squeezings)
        ]
        + instructions_from_decomposition(decomposition)
    )

    simulator = pq.GaussianSimulator(d=d)

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


def test_instructions_from_decomposition_using_piquasso_GaussianSimulator_with_context(
    dummy_unitary,
):
    d = 3
    U = dummy_unitary(d)

    decomposition = clements(U, connector=pq.NumpyConnector())

    squeezings = [0.1, 0.2, 0.3]

    with pq.Program() as program_with_interferometer:
        for mode, r in enumerate(squeezings):
            pq.Q(mode) | pq.Squeezing(r)

        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        for mode, r in enumerate(squeezings):
            pq.Q(mode) | pq.Squeezing(r)

        program_with_decomposition.instructions.extend(
            instructions_from_decomposition(decomposition)
        )

    simulator = pq.GaussianSimulator(d=d)

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


@pytest.mark.monkey
def test_instructions_from_decomposition_using_piquasso_GaussianSimulator_random(
    dummy_unitary, generate_gaussian_transform
):
    d = 3
    U = dummy_unitary(d)

    decomposition = clements(U, connector=pq.NumpyConnector())

    gaussian_transform = generate_gaussian_transform(d)

    with pq.Program() as program_with_interferometer:
        pq.Q() | gaussian_transform
        pq.Q() | pq.Interferometer(matrix=U)

    with pq.Program() as program_with_decomposition:
        pq.Q() | gaussian_transform
        program_with_decomposition.instructions.extend(
            instructions_from_decomposition(decomposition)
        )

    simulator = pq.GaussianSimulator(d=d)

    state_with_interferometer = simulator.execute(program_with_interferometer).state
    state_with_decomposition = simulator.execute(program_with_decomposition).state

    assert state_with_interferometer == state_with_decomposition


connectors = [pytest.param(pq.NumpyConnector(), id="numpy")]

try:
    import tensorflow as tf  # noqa: F811, F401
    connectors.append(pytest.param(pq.TensorflowConnector(), id="tensorflow"))
except ImportError:
    pass

if JAX_AVAILABLE:
    connectors.append(pytest.param(pq.JaxConnector(), id="jax"))


@pytest.mark.parametrize("connector", connectors)
def test_clements_decomposition_roundtrip(connector, dummy_unitary):
    """Test that the Clements decomposition roundtrips correctly."""
    if isinstance(connector, pq.TensorflowConnector) and sys.version_info >= (3, 13):
        pytest.skip("TensorFlow not supported on Python 3.13+")
    
    d = 5
    U = dummy_unitary(d)

    decomposition = clements(U, connector)

    new_U = inverse_clements(decomposition, connector, dtype=U.dtype)

    assert np.allclose(U, new_U)


@pytest.mark.parametrize("connector", connectors)
def test_clements_decomposition_roundtrip_with_weights(connector, dummy_unitary):
    """Test that the Clements decomposition roundtrips correctly with weights."""
    if isinstance(connector, pq.TensorflowConnector) and sys.version_info >= (3, 13):
        pytest.skip("TensorFlow not supported on Python 3.13+")
    
    d = 6
    U = dummy_unitary(d)

    decomposition = clements(U, connector)

    weights = get_weights_from_decomposition(decomposition, d, connector)

    new_decomposition = get_decomposition_from_weights(weights, d, connector)

    assert decomposition == new_decomposition

    new_U = inverse_clements(new_decomposition, connector, dtype=U.dtype)

    assert np.allclose(U, new_U)


@pytest.mark.parametrize("connector", connectors)
def test_weigths_to_interferometer_roundtrip(connector, dummy_unitary):
    """Test that weights to interferometer conversion roundtrips correctly."""
    if isinstance(connector, pq.TensorflowConnector) and sys.version_info >= (3, 13):
        pytest.skip("TensorFlow not supported on Python 3.13+")
    
    d = 6
    U = connector.np.array(dummy_unitary(d))

    weights = get_weights_from_interferometer(U, connector)

    new_U = get_interferometer_from_weights(weights, d, connector, dtype=U.dtype)

    assert np.allclose(U, new_U)


@pytest.mark.tensorflow
def test_clements_decomposition_with_TensorflowConnector(dummy_unitary):
    """Test Clements decomposition with TensorFlow connector."""
    if sys.version_info >= (3, 13):
        pytest.skip("TensorFlow not supported on Python 3.13+")
    
    U = tf.Variable(dummy_unitary(2))

    with tf.GradientTape() as tape:
        decomposition = clements(U, connector=pq.TensorflowConnector())

    assert len(decomposition.beamsplitters) == 1
    assert len(decomposition.phaseshifters) == 2

    first_beamsplitter_theta = decomposition.beamsplitters[0].params[0]

    assert np.isclose(first_beamsplitter_theta, np.arctan(np.abs(U[1, 0] / U[0, 0])))

    gradient = tape.gradient(first_beamsplitter_theta, U)

    assert np.allclose(
        gradient,
        np.array(
            [
                [-np.abs(U[1, 0]) * U[0, 0] / np.abs(U[0, 0]), 0.0],
                [np.abs(U[0, 0]) * U[1, 0] / np.abs(U[1, 0]), 0.0],
            ]
        ),
    )


@pytest.mark.tensorflow
def test_clements_decomposition_with_TensorflowConnector_from_weights(dummy_unitary):
    """Test Clements decomposition with TensorFlow connector from weights."""
    if sys.version_info >= (3, 13):
        pytest.skip("TensorFlow not supported on Python 3.13+")
    
    d = 2
    U = dummy_unitary(d)

    weights = tf.Variable(get_weights_from_interferometer(U, pq.NumpyConnector()))

    with tf.GradientTape() as tape:
        decomposition = get_decomposition_from_weights(
            weights, d=d, connector=pq.TensorflowConnector()
        )

    assert len(decomposition.beamsplitters) == 1
    assert len(decomposition.phaseshifters) == 2

    first_beamsplitter_theta = decomposition.beamsplitters[0].params[0]

    assert np.isclose(first_beamsplitter_theta, np.arctan(np.abs(U[1, 0] / U[0, 0])))

    gradient = tape.gradient(first_beamsplitter_theta, weights)

    assert np.allclose(gradient, [1.0, 0.0, 0.0, 0.0])


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_clements_decomposition_with_JaxConnector(dummy_unitary):
    """Test Clements decomposition with JAX connector."""
    def get_first_beamsplitter_theta(U):
        decomposition = clements(U, connector=pq.JaxConnector())

        assert len(decomposition.beamsplitters) == 1
        assert len(decomposition.phaseshifters) == 2

        first_beamsplitter_theta = decomposition.beamsplitters[0].params[0]

        return first_beamsplitter_theta

    U = dummy_unitary(2)

    first_beamsplitter_theta = get_first_beamsplitter_theta(U)

    assert np.isclose(first_beamsplitter_theta, np.arctan(np.abs(U[1, 0] / U[0, 0])))

    grad_first_beamsplitter_theta = grad(get_first_beamsplitter_theta)

    gradient = grad_first_beamsplitter_theta(U)

    # NOTE: There is a difference between the differential with respect to a complex
    # variable between Tensorflow and JAX, namely, that the gradients in the two
    # frameworks are conjugate to each other.
    assert np.allclose(
        gradient,
        np.conj(
            np.array(
                [
                    [-np.abs(U[1, 0]) * U[0, 0] / np.abs(U[0, 0]), 0.0],
                    [np.abs(U[0, 0]) * U[1, 0] / np.abs(U[1, 0]), 0.0],
                ]
            )
        ),
    )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
def test_clements_decomposition_with_JaxConnector_from_weights(dummy_unitary):
    """Test Clements decomposition with JAX connector from weights."""
    d = 2
    U = dummy_unitary(d)

    weights = get_weights_from_interferometer(U, pq.NumpyConnector())

    connector = pq.JaxConnector()

    def get_first_beamsplitter_theta(weights):
        decomposition = get_decomposition_from_weights(
            weights, d=d, connector=connector
        )

        first_beamsplitter_theta = decomposition.beamsplitters[0].params[0]

        return first_beamsplitter_theta

    first_beamsplitter_theta = get_first_beamsplitter_theta(weights)

    assert np.isclose(first_beamsplitter_theta, np.arctan(np.abs(U[1, 0] / U[0, 0])))

    grad_first_beamsplitter_theta = grad(get_first_beamsplitter_theta)

    gradient = grad_first_beamsplitter_theta(weights)

    assert np.allclose(
        gradient,
        [1.0, 0.0, 0.0, 0.0],
    )
