#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

from piquasso._backends.fock.pure.simulator import PureFockSimulator

from .calculator import TensorflowCalculator


class TensorflowPureFockSimulator(PureFockSimulator):
    """Performs photonic simulations using Fock representation with pure states.

    This simulator is similar to
    :class:`~piquasso._backends.fock.pure.simulator.PureFockSimulator`, but it
    calculates with Tensorflow to enable calculating the gradient.

    The simulation (when executed) results in an instance of
    :class:`~piquasso._backends.fock.pure.state.PureFockState`.

    Note:
        Non-deterministic operations like
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement`  are
        non-differentiable, please use a deterministic attribute of the resulting state
        instead.

    Example usage::

        import tensorflow as tf

        alpha = tf.Variable(0.43)

        simulator = pq.TensorflowPureFockSimulator(d=1)

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()

            pq.Q(0) | pq.Displacement(alpha=alpha)

        with tf.GradientTape() as tape:
            state = simulator.execute(program).state

            mean = state.mean_photon_number()

        gradient = tape.gradient(mean, [alpha])

    Supported preparations:
        :class:`~piquasso.instructions.preparations.Vacuum`,
        :class:`~piquasso.instructions.preparations.Create`,
        :class:`~piquasso.instructions.preparations.Annihilate`,
        :class:`~piquasso.instructions.preparations.StateVector`.

    Supported gates:
        :class:`~piquasso.instructions.gates.Interferometer`,
        :class:`~piquasso.instructions.gates.Beamsplitter`,
        :class:`~piquasso.instructions.gates.Phaseshifter`,
        :class:`~piquasso.instructions.gates.MachZehnder`,
        :class:`~piquasso.instructions.gates.Fourier`,
        :class:`~piquasso.instructions.gates.Kerr`,
        :class:`~piquasso.instructions.gates.CrossKerr`,
        :class:`~piquasso.instructions.gates.CubicPhase`,
        :class:`~piquasso.instructions.gates.GaussianTransform`,
        :class:`~piquasso.instructions.gates.Squeezing`,
        :class:`~piquasso.instructions.gates.QuadraticPhase`,
        :class:`~piquasso.instructions.gates.Squeezing2`,
        :class:`~piquasso.instructions.gates.ControlledX`,
        :class:`~piquasso.instructions.gates.ControlledZ`,
        :class:`~piquasso.instructions.gates.Displacement`,
        :class:`~piquasso.instructions.gates.PositionDisplacement`,
        :class:`~piquasso.instructions.gates.MomentumDisplacement`.

    Supported measurements:
        :class:`~piquasso.instructions.measurements.ParticleNumberMeasurement` (non-differentiable).

    Supported channels:
        :class:`~piquasso.instructions.channels.Attenuator` (non-differentiable).
    """  # noqa: E501

    _calculator_class = TensorflowCalculator
