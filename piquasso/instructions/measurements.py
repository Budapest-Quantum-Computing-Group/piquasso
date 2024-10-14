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

"""
Measurements
============

All the built-in measurement instructions in Piquasso. Measurement instructions should
be placed at the end of the Piquasso program.

.. note::

   When multiple shots are specified and the evolution of the choses simulated state is
   possible, the measurement outcome corresponding to the last shot is used to evolve
   the state.

"""

from typing import Tuple

import numpy as np

from piquasso.api.exceptions import InvalidParameter
from piquasso.api.instruction import Measurement
from piquasso._math.linalg import is_positive_semidefinite
from piquasso._math.symplectic import symplectic_form


class ParticleNumberMeasurement(Measurement):
    r"""Particle number measurement.

    A non-Gaussian projective measurement with the probability density given by

    .. math::
        p(n) = \operatorname{Tr} \left [ \rho | n \rangle \langle n | \right ]


    The generated samples are non-negative integer values corresponding to the detected
    photon number.

    .. note::

        When used with :class:`~piquasso._simulators.gaussian.state.GaussianState`, the
        state is not evolved, since that would be non-Gaussian.

    """

    def __init__(self) -> None:
        super().__init__()


class ThresholdMeasurement(Measurement):
    """Threshold measurement.

    Similar to :class:`ParticleNumberMeasurement`, but only measuring whether or not
    the measured mode contains any photon.

    The generated samples contain :math:`0` or :math:`1`, where :math:`0` corresponds to
    no photon being detected, and :math:`1` corresponds to detection of at least one
    photon.

    """

    def __init__(self) -> None:
        super().__init__()


class GeneraldyneMeasurement(Measurement):
    r"""General-dyne measurement.

    The probability density is given by

    .. math::
        p(r_m) = \frac{
            \exp \left (
                (r_m - r)^T
                \frac{1}{\sigma + \sigma_m}
                (r_m - r)
            \right )
        }{
            \pi^d \sqrt{ \operatorname{det} (\sigma + \sigma_m) }
        },

    where :math:`r_m \in \mathbb{C}^d`, :math:`\sigma` is the covariance matrix of the
    current state, :math:`r \in \mathbb{C}^d` is the first moment of the current state,
    and :math:`\sigma_m` is the covariance corresponding to a non-displaced Gaussian
    state characterizing the general-dyne detection. Notably, the heterodyne detection
    would correspond to a non-displaced Gaussian state with covariance
    :math:`\sigma_m = I_{d \times d}`.
    """

    def __init__(self, detection_covariance: np.ndarray) -> None:
        r"""
        Args:
            detection_covariance (numpy.ndarray):
                A :math:`2\times2` symplectic matrix corresponding to a purely
                quadratic Hamiltonian.

        Raises:
            InvalidParameter:
                When the detection covariance does not satisfy the Robertson-Schrödinger
                uncertainty relation.
        """

        if not is_positive_semidefinite(detection_covariance + 1j * symplectic_form(1)):
            raise InvalidParameter(
                "The parameter 'detection_covariance' is invalid, since it doesn't "
                "fulfill the Robertson-Schrödinger uncertainty relation."
            )

        super().__init__(
            params=dict(
                detection_covariance=detection_covariance,
            )
        )


class HomodyneMeasurement(Measurement):
    r"""Homodyne measurement.

    Corresponds to measurement of the quadrature operator

    .. math::
        \hat{x}_{\phi} = \cos \phi \hat{x} + \sin \phi \hat{p}

    with outcome probability density given by

    .. math::
        p(x_{\phi}) = \langle x_{\phi} | \rho | x_{\phi} \rangle,

    where :math:`x_{\phi}` correspond to the eigenvalues of :math:`\hat{x}_{\phi}`.

    In optical setups, the measurement is performed by mixing the state :math:`\rho`
    with a strong coherent state :math:`| \alpha \rangle`, where :math:`\alpha >> 1`,
    then subtracting the detected intensities of the two outputs.
    The mixing is performed with a 50:50 beamsplitter.
    """

    def __init__(self, phi: float = 0.0, z: float = 1e-4) -> None:
        """
        Args:
            phi (float): Phase space rotation angle.
            z (float):
                Squeezing amplitude. In the limit of `z` going to infinity one would
                recover the pure homodyne measurement in the so-called strong oscillator
                limit. Conversely, setting `z = 1` would correspond to
                :class:`HeterodyneMeasurement`.
        """

        super().__init__(
            params=dict(
                phi=phi,
                z=z,
            ),
            extra_params=dict(
                detection_covariance=np.array(
                    [
                        [z**2, 0],
                        [0, (1 / z) ** 2],
                    ]
                ),
            ),
        )


class HeterodyneMeasurement(Measurement):
    r"""Heterodyne measurement.

    The probability density is given by

    .. math::
        p(x_{\phi}) = \frac{1}{\pi} \operatorname{Tr} (
            \rho | \alpha \rangle \langle \alpha |
        ).

    In optical setups, the measurement is performed by mixing the state :math:`\rho`
    with a vacuum state :math:`| 0 \rangle`, then subtracting the detected intensities
    of the two outputs.
    The mixing is performed with a 50:50 beamsplitter.
    """

    def __init__(self) -> None:
        super().__init__(
            extra_params=dict(
                detection_covariance=np.identity(2),
            ),
        )


class PostSelectPhotons(Measurement):
    """Post-selection on detected photon numbers.

    Example usage::

        with pq.Program() as program:
            pq.Q(all) | pq.StateVector([0, 1, 0]) * np.sqrt(0.2)
            pq.Q(all) | pq.StateVector([1, 1, 0]) * np.sqrt(0.3)
            pq.Q(all) | pq.StateVector([2, 1, 0]) * np.sqrt(0.5)

            pq.Q(0) | pq.Phaseshifter(np.pi)

            pq.Q(1, 2) | pq.Beamsplitter(np.pi / 8)
            pq.Q(0, 1) | pq.Beamsplitter(1.1437)
            pq.Q(1, 2) | pq.Beamsplitter(-np.pi / 8)

            pq.Q(all) | pq.PostSelectPhotons(
                postselect_modes=(1, 2), photon_counts=(1, 0)
            )

        simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=4))

        state = simulator.execute(program).state


    Note:
        The resulting state is not normalized. To normalize it, use
        :meth:`~piquasso._simulators.fock.pure.state.PureFockState.normalize`. Moreover,
        it is a state over the remaining modes, i.e., if `k` modes are post-selected,
        then the resulting state will be defined on a system with `k` modes less.
    """

    def __init__(
        self, postselect_modes: Tuple[int, ...], photon_counts: Tuple[int, ...]
    ):
        """
        Args:
            postselect_modes (Tuple[int, ...]): The modes to post-select on.
            photon_counts (Tuple[int, ...]):
                The desired photon numbers on the specified modes.
        """

        super().__init__(
            params=dict(postselect_modes=postselect_modes, photon_counts=photon_counts)
        )


class ImperfectPostSelectPhotons(Measurement):
    r"""Post-selection on detected photon numbers.

    Example usage::

        P = np.array(
            [
                [1.0, 0.1, 0.2],
                [0.0, 0.9, 0.2],
                [0.0, 0.0, 0.6],
            ],
        )

        with pq.Program() as program:
            pq.Q(all) | pq.StateVector([0, 1, 0]) * np.sqrt(0.2)
            pq.Q(all) | pq.StateVector([1, 1, 0]) * np.sqrt(0.3)
            pq.Q(all) | pq.StateVector([2, 1, 0]) * np.sqrt(0.5)

            pq.Q(0) | pq.Phaseshifter(np.pi)

            pq.Q(1, 2) | pq.Beamsplitter(np.pi / 8)
            pq.Q(0, 1) | pq.Beamsplitter(1.1437)
            pq.Q(1, 2) | pq.Beamsplitter(-np.pi / 8)

            pq.Q(all) | pq.ImperfectPostSelectPhotons(
                postselect_modes=(1, 2), photon_counts=(1, 0),
                detector_efficiency_matrix=P,
            )

        simulator = pq.PureFockSimulator(d=3, config=pq.Config(cutoff=4))

        state = simulator.execute(program).state


    The detector efficiency matrix :math:`P` is defined elementwise as

    .. math::

        P_{n m} = p(\text{detected photon count } = n| \text{actual photon count } = m)

    Note:
        The resulting state is not normalized. To normalize it, use
        :meth:`~piquasso._simulators.fock.pure.state.PureFockState.normalize`. Moreover,
        it is a state over the remaining modes, i.e., if `k` modes are post-selected,
        then the resulting state will be defined on a system with `k` modes less.
    """

    def __init__(
        self,
        postselect_modes: Tuple[int, ...],
        photon_counts: Tuple[int, ...],
        detector_efficiency_matrix: "np.ndarray",
    ):
        """_summary_

        Args:
            postselect_modes (Tuple[int, ...]): The modes to post-select on.
            photon_counts (Tuple[int, ...]):
                The desired photon numbers on the specified modes.
            detector_efficiency_matrix (np.ndarray): Detector efficiency matrix.
        """
        super().__init__(
            params=dict(
                postselect_modes=postselect_modes,
                photon_counts=photon_counts,
                detector_efficiency_matrix=detector_efficiency_matrix,
            )
        )
