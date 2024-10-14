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
Channels
========

The built-in channel instructions in Piquasso.
"""

import numpy as np

from piquasso._math.linalg import is_positive_semidefinite, is_real_2n_by_2n
from piquasso._math.symplectic import symplectic_form
from piquasso._math.validations import all_in_interval

from piquasso.api.exceptions import InvalidParameter
from piquasso.api.instruction import Gate


class DeterministicGaussianChannel(Gate):
    r"""Deterministic Gaussian channel.

    It is a CP (completely positive) map between Gaussian states which is characterized
    by the matrices :math:`X, Y \in \mathbb{R}^{2n \times 2n}` acting on the mean vector
    and covariance matrix with the mapping

    .. math::
        \mu &\mapsto X \mu \\
        \sigma &\mapsto X \sigma X^T + Y.

    The matrices :math:`X` and :math:`Y` should satisfy the inequality

    .. math::
        Y + i \Omega \geq i X \Omega X^T.

    Note:
        The matrix :math:`Y` is dependent on :math:`\hbar`, but the value of
        :math:`\hbar` is specified later when executed by a simulator. The parameter
        `Y` should be specified keeping in mind that it will automatically be scaled
        with :math:`\hbar` during execution.

    Note:
        Currently, this instruction can only be used along with
        :class:`~piquasso._simulators.gaussian.state.GaussianState`.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Args:
            X (numpy.ndarray):
                Transformation matrix on the quadrature vectors in `xpxp` order.
            Y (numpy.ndarray):
                The additive noise contributing to the covariance matrix in `xpxp`
                order.

        Raises:
            InvalidParameter: If the specified 'X' and/or 'Y' matrices are invalid.
        """

        if not is_real_2n_by_2n(X):
            raise InvalidParameter(
                f"The parameter 'X' must be a real 2n-by-2n matrix: X={X}"
            )

        if not is_real_2n_by_2n(Y):
            raise InvalidParameter(
                f"The parameter 'Y' must be a real 2n-by-2n matrix: Y={Y}"
            )

        if X.shape != Y.shape:
            raise InvalidParameter(
                f"The shape of matrices 'X' and 'Y' should be equal: "
                f"X.shape={X.shape}, Y.shape={Y.shape}"
            )

        omega = symplectic_form(len(X) // 2)

        if not is_positive_semidefinite(
            Y - 1j * omega - 1j * X @ omega @ X.transpose()
        ):
            raise InvalidParameter(
                "The matrices 'X' and 'Y' does not satisfy the inequality "
                "corresponding to Gaussian channels."
            )

        super().__init__(params=dict(X=X, Y=Y))


class Attenuator(Gate):
    r"""The attenuator channel.

    It reduces the first moments' amplitude by :math:`\cos \theta` by mixing the input
    state with a thermal state using a beamsplitter.

    It can also be characterized as a deterministic Gaussian channel with mappings

    .. math::
        X &= \cos \theta I_{2 \times 2} \\
        Y &= ( \sin \theta )^2 (2 \bar{N} + 1) I_{2 \times 2},

    where :math:`\theta \in [0, 2 \pi)` is the beampsplitters' mixing angle and
    :math:`\bar{N} \in \mathbb{R}^{+}`. :math:`\bar{N}` is the mean number of thermal
    excitations of the system interacting with the environment.

    Note:
        Currently, this instruction can only be used along with
        :class:`~piquasso._simulators.gaussian.state.GaussianState`.
    """

    def __init__(self, theta: float, mean_thermal_excitation: float = 0) -> None:
        """
        Args:
            theta (float): The mixing angle.
            mean_thermal_excitation (int):
                Mean number of thermal excitations of the system interacting with the
                environment.

        Raises:
            InvalidParameter: If the specified mean thermal excitation is not positive.
        """

        if mean_thermal_excitation < 0:
            raise InvalidParameter(
                "The parameter 'mean_thermal_excitation' must be a positive real "
                f"number: mean_thermal_excitation={mean_thermal_excitation}"
            )

        X = np.cos(theta) * np.identity(2)

        Y = (np.sin(theta) ** 2) * (2 * mean_thermal_excitation + 1) * np.identity(2)

        super().__init__(
            params=dict(theta=theta, mean_thermal_excitation=mean_thermal_excitation),
            extra_params=dict(X=X, Y=Y),
        )


class Loss(Gate):
    r"""Applies a loss channel to the state.

    The transmissivity is defined by :math:`t = \cos \theta`, where :math:`\theta` is
    the beamsplitter parameter and the angle between the initial and resulting state.
    Considering only one particle, :math:`1-t^2` is the probability of losing this
    particle.

    Note:
        Currently, this instruction can only be used along with
        :class:`~piquasso._simulators.sampling.simulator.SamplingSimulator`.

    Note:
        The parameter `transmissivity` is usually called `transmittance`.

    """

    NUMBER_OF_MODES = 1

    def __init__(self, transmissivity: np.ndarray) -> None:
        """
        Args:
            transmissivity (numpy.ndarray): The transmissivity.
        """
        super().__init__(params=dict(transmissivity=transmissivity))


class LossyInterferometer(Gate):
    """Applies a lossy interferometer (specified by a matrix) to the state.

    The lossy interferometer matrix :math:`A` should have singular values in the
    interval :math:`[0, 1]`.

    Applying :class:`LossyInterferometer` with parameter :math:`A` on all modes is
    equivalent to the following::

        pq.Q() | pq.Interferometer(W) | pq.Loss(np.diag(D)) | pq.Interferometer(V)

    where :math:`A = V D W`, :math:`V, W` are unitary matrices and :math:`D` a
    diagonal matrix. The diagonal entries in :math:`D` are called the singular values.

    Note that the :math:`W` and :math:`V` interferometer matrices and the vector
    :math:`D` are the results of the singular value decomposition (SVD) of the
    (possibly lossy) interferometer A.

    Note:
        Currently, this instruction can only be used along with
        :class:`~piquasso._simulators.sampling.simulator.SamplingSimulator`.

    Raises:
        InvalidParameter:
            When the singular values are not in the interval :math:`[0, 1]`.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Args:
            matrix (numpy.ndarray): The matrix representing the lossy interferometer.
        """

        _, singular_values, _ = np.linalg.svd(matrix)

        if not all_in_interval(singular_values, lower=0.0, upper=1.0):
            raise InvalidParameter(
                "The specified lossy interferometer matrix has singular values outside "
                f"of the interval [0, 1]: 'singular_values={singular_values}'"
            )

        super().__init__(
            params=dict(matrix=matrix),
        )
