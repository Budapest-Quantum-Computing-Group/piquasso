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

import numpy as np

from piquasso._math.linalg import is_positive_semidefinite, is_real_2n_by_2n
from piquasso._math.symplectic import symplectic_form

from piquasso.core import _mixins

from piquasso.api.errors import InvalidParameter
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
        :class:`~piquasso._backends.gaussian.state.GaussianState`.
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
        :class:`~piquasso._backends.gaussian.state.GaussianState`.
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


class Loss(Gate, _mixins.ScalingMixin):
    """Applies a loss channel to the state.

    Note:
        Currently, this instruction can only be used along with
        :class:`~piquasso._backends.sampling.state.SamplingState`.
    """

    def __init__(self, transmissivity: np.ndarray) -> None:
        """
        Args:
            transmissivity (numpy.ndarray): The transmissivity array.
        """
        super().__init__(
            params=dict(transmissivity=transmissivity),
            extra_params=dict(
                transmissivity=np.atleast_1d(transmissivity),
            ),
        )

    def _autoscale(self) -> None:
        transmissivity = self._extra_params["transmissivity"]
        if transmissivity is None or len(self.modes) == len(transmissivity):
            pass
        elif len(transmissivity) == 1:
            self._extra_params["transmissivity"] = np.array(
                [transmissivity[0]] * len(self.modes),
                dtype=complex,
            )
        else:
            raise InvalidParameter(
                f"The channel {self} is not applicable to modes {self.modes} with the "
                "specified parameters."
            )
