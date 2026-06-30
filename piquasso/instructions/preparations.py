#
# Copyright 2021-2026 Budapest Quantum Computing Group
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
Preparations
============

The built-in preparation instructions in Piquasso. Preparation instructions should
be placed at the beginning of the Piquasso program.
"""


from typing import Dict, Iterable, Tuple, Optional, Union
import warnings

import numpy as np

from piquasso.core import _mixins
from piquasso.api.instruction import Preparation
from piquasso.api.exceptions import InvalidParameter, InvalidState
from piquasso.api.connector import BaseConnector

from piquasso._math.validations import all_natural, all_real_and_positive


class Vacuum(Preparation):
    r"""Prepare the system in a vacuum state.

    Example usage::

        with pq.Program() as program:
            pq.Q() | pq.Vacuum()
            ...

    Note:
        This operation can only be used for all modes and is only practical to be used
        at the beginning of a program declaration.
    """

    def __init__(self) -> None:
        super().__init__()


class Mean(Preparation):
    r"""Set the first canonical moment of the state.

    Example usage::

        with pq.Program() as program:
            pq.Q() | pq.Mean(
                mean=np.array(...)
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._simulators.gaussian.state.GaussianState`.

    Note:
        The mean vector is dependent on :math:`\hbar`, but the value of :math:`\hbar`
        is specified later when executed by a simulator. The parameter `mean` should be
        specified keeping in mind that it will automatically be scaled with
        :math:`\hbar` during execution.
    """

    def __init__(self, mean: np.ndarray) -> None:
        """
        Args:
            mean (numpy.ndarray):
                The vector of the first canonical moments in `xpxp`-ordering.
        """
        super().__init__(params=dict(mean=mean))


class Covariance(Preparation):
    r"""Sets the covariance matrix of the state.

    Example usage::

        with pq.Program() as program:
            pq.Q() | pq.Covariance(
                cov=np.array(...)
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._simulators.gaussian.state.GaussianState`.

    Note:
        The covariance matrix is dependent on :math:`\hbar`, but the value of
        :math:`\hbar` is specified later when executed by a simulator. The parameter
        `cov` should be specified keeping in mind that it will automatically be scaled
        with :math:`\hbar` during execution.
    """

    def __init__(self, cov: np.ndarray) -> None:
        """
        Args:
            cov (numpy.ndarray): The covariance matrix in `xpxp`-ordering.
        """

        super().__init__(params=dict(cov=cov))


class Thermal(Preparation):
    r"""Prepares a thermal state.

    Example usage::

        with pq.Program() as program:
            pq.Q(0, 1) | pq.Thermal([0.5, 1.5])
            ...

    The thermal state is defined by

    .. math::
        \rho = \sum_{\vec{n} = 0}^\infty
            \frac{\overline{n}^{\vec{n}}}{(1 + \overline{n})^{\vec{n} + 1}}
            | \vec{n} \rangle \langle \vec{n} |,

    where :math:`\overline{n} \in \mathbb{R}^d` is the vector of mean photon numbers.
    The power on vectors is defined as

    .. math::
        \vec{a}^{\vec{b}} = \prod_{i=1}^d a_i^{b_i}.

    In terms of Gaussian states, the mean vector and covariance matrix is

    .. math::
        \mu &= 0_{2d} \\
        \sigma &= \hbar (
            2 \operatorname{diag}(\operatorname{repeat}
                (\overline{n}, 2)) + I_{2d \times 2d}
        ).

    Can only be applied to the following states:
    :class:`~piquasso._simulators.gaussian.state.GaussianState`.
    """

    def __init__(self, mean_photon_numbers: Iterable[float]) -> None:
        """
        Args:
            mean_photon_numbers (Iterable[float]): The sequence of mean photon numbers.

        Raises:
            InvalidParameter: If the mean photon numbers are not positive real numbers.
        """

        super().__init__(params=dict(mean_photon_numbers=mean_photon_numbers))

    def _validate(self, connector: BaseConnector) -> None:
        mean_photon_numbers = self.params["mean_photon_numbers"]

        if connector.is_abstract(mean_photon_numbers):
            return

        if not all_real_and_positive(mean_photon_numbers):
            raise InvalidParameter(
                "The mean photon numbers must be positive real numbers: "
                f"mean_photon_numbers: {mean_photon_numbers}"
            )

    def _get_computed_params(self, connector: BaseConnector) -> dict:
        np = connector.np
        mean_photon_numbers = self.params["mean_photon_numbers"]

        cov = np.diag(2 * np.repeat(np.array(mean_photon_numbers), 2) + 1)

        return dict(cov=cov)


class NumberState(Preparation, _mixins.WeightMixin):
    r"""State preparation with Fock basis vectors.

    Example usage::

        with pq.Program() as program:
            pq.Q() | (
                0.3 * pq.NumberState([2, 1, 0, 3])
                + 0.2 * pq.NumberState([1, 1, 2, 3])
                ...
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._simulators.fock.pure.state.PureFockState` and
    :class:`~piquasso._simulators.sampling.state.SamplingState`.
    """

    def __init__(
        self,
        occupation_numbers: Iterable[int],
        coefficient: complex = 1.0,
    ) -> None:
        """
        Args:
            occupation_numbers (Iterable[int]): The occupation numbers.
            coefficient (complex, optional):
                The coefficient of the occupation number. Defaults to :math:`1.0`.

        Raises:
            InvalidState:
                If the specified occupation numbers are not all natural numbers.
        """

        if not all_natural(occupation_numbers):
            raise InvalidState(
                f"Invalid occupation numbers: "
                f"occupation_numbers={occupation_numbers}\n"
                "Occupation numbers must contain non-negative integers."
            )

        super().__init__(
            params=dict(
                occupation_numbers=tuple(occupation_numbers),
                coefficient=coefficient,
            ),
        )

    def __add__(self, other):
        if isinstance(other, NumberState):
            if self.params["occupation_numbers"] == other.params["occupation_numbers"]:
                new_coefficient = (
                    self.params["coefficient"] + other.params["coefficient"]
                )
                return NumberState(
                    occupation_numbers=self.params["occupation_numbers"],
                    coefficient=new_coefficient,
                )
            else:
                fock_amplitude_map = {
                    self.params["occupation_numbers"]: self.params["coefficient"],
                    other.params["occupation_numbers"]: other.params["coefficient"],
                }
                return FockStateVector(fock_amplitude_map=fock_amplitude_map)
        elif isinstance(other, FockStateVector):
            if self.params["occupation_numbers"] in other.params["fock_amplitude_map"]:
                new_coefficient = (
                    self.params["coefficient"]
                    + other.params["fock_amplitude_map"][
                        self.params["occupation_numbers"]
                    ]
                )
                fock_amplitude_map = {
                    **other.params["fock_amplitude_map"],
                    self.params["occupation_numbers"]: new_coefficient,
                }
            else:
                fock_amplitude_map = {
                    self.params["occupation_numbers"]: self.params["coefficient"],
                    **other.params["fock_amplitude_map"],
                }

            return FockStateVector(fock_amplitude_map=fock_amplitude_map)
        else:
            return NotImplemented


class DistinguishableNumberState(Preparation):
    r"""State preparation with labelled photons in spatial number states.

    The occupation numbers specify the spatial input modes. The optional
    ``particle_overlap`` specifies the internal-state Gram matrix of the labelled
    photons.

    Example usage with uniform partial distinguishability::

        with pq.Program() as program:
            pq.Q() | pq.DistinguishableNumberState(
                [2, 1, 0, 3],
                particle_overlap=0.8,
            )
            ...

    Example usage with a full particle-overlap matrix::

        G = np.array(
            [
                [1.0, 0.8, 0.7],
                [0.8, 1.0, 0.6],
                [0.7, 0.6, 1.0],
            ],
            dtype=complex,
        )

        with pq.Program() as program:
            pq.Q() | pq.DistinguishableNumberState(
                [1, 1, 1, 0],
                particle_overlap=G,
            )
            ...

    The photon ordering is induced by the occupation numbers in increasing mode
    order. For example, ``occupation_numbers=[2, 0, 1]`` corresponds to labelled
    photon input modes ``[0, 0, 2]``. Therefore, a full overlap matrix ``G`` must
    satisfy ``G[i, j] = <phi_i | phi_j>`` with respect to this ordering.

    A scalar ``particle_overlap=lambda`` means real uniform amplitude overlap,

    .. math::

        G_{ii} = 1, \qquad G_{ij} = \lambda \quad i \neq j.

    A matrix ``particle_overlap=G`` must be a normalized positive semidefinite
    Gram matrix,

    .. math::

        G = G^\dagger, \qquad G \succeq 0, \qquad G_{ii}=1.

    Complex-valued overlaps should be specified using the full Gram matrix form,
    so that Hermiticity is explicit.

    The scalar uniform-overlap algorithms are typically faster than the general
    Gram-matrix algorithms.

    Can only be applied to the following states:
    :class:`~piquasso._simulators.sampling.state.SamplingState`.
    """

    def __init__(
        self,
        occupation_numbers: Iterable[int],
        particle_overlap: Union[float, np.ndarray] = 0.0,
    ) -> None:
        """
        Args:
            occupation_numbers (Iterable[int]):
                The spatial occupation numbers.

            particle_overlap (Union[float, np.ndarray]):
                Either a scalar uniform amplitude overlap or an ``n x n``
                particle-overlap Gram matrix, where
                ``n = sum(occupation_numbers)``.

                A scalar value means uniform pairwise overlap between every
                pair of labelled photons.

                A matrix value means ``particle_overlap[i, j]`` is the
                internal-state overlap of photons ``i`` and ``j`` in the
                ordering induced by ``occupation_numbers``.


        Raises:
            InvalidState:
                If the occupation numbers are invalid, or if the particle
                overlap is not a valid scalar overlap or Gram matrix.
        """

        super().__init__(
            params=dict(
                occupation_numbers=tuple(occupation_numbers),
                particle_overlap=particle_overlap,
            ),
        )

    def _validate(self, connector: BaseConnector) -> None:
        np = connector.np

        occupation_numbers = self.params["occupation_numbers"]
        particle_overlap = self.params["particle_overlap"]

        if not all_natural(occupation_numbers):
            raise InvalidState(
                f"Invalid occupation numbers: "
                f"occupation_numbers={occupation_numbers}\n"
                "Occupation numbers must contain non-negative integers."
            )

        if connector.is_abstract(particle_overlap):
            return

        if np.isscalar(particle_overlap):
            if particle_overlap < 0.0 or particle_overlap > 1.0:
                raise InvalidState(
                    f"Invalid particle overlap: "
                    f"particle_overlap={particle_overlap}\n"
                    "Particle overlap must be in the range [0, 1]."
                )

        else:
            number_of_particles = sum(occupation_numbers)
            expected_shape = (number_of_particles, number_of_particles)

            if particle_overlap.shape != expected_shape:
                raise InvalidState(
                    f"Invalid particle overlap matrix shape: "
                    f"particle_overlap.shape={particle_overlap.shape}\n"
                    f"For occupation_numbers={occupation_numbers}, expected shape "
                    f"{expected_shape}."
                )

            if not np.allclose(particle_overlap, particle_overlap.conj().T):
                raise InvalidState(
                    f"Invalid particle overlap matrix: "
                    f"particle_overlap={particle_overlap}\n"
                    "Particle overlap matrix must be Hermitian."
                )

            if not np.allclose(np.diag(particle_overlap), 1.0):
                raise InvalidState(
                    f"Invalid particle overlap matrix diagonal: "
                    f"diag={np.diag(particle_overlap)}\n"
                    "Particle overlap matrix must have ones on its diagonal."
                )

            eigenvalues = np.linalg.eigvalsh(particle_overlap)
            if np.any(eigenvalues < -1e-10):
                raise InvalidState(
                    f"Invalid particle overlap matrix eigenvalues: "
                    f"eigenvalues={eigenvalues}\n"
                    "Particle overlap matrix must be positive semidefinite."
                )


class FockStateVector(Preparation, _mixins.WeightMixin):
    r"""State preparation with Fock basis vectors.

    Example usage::

        with pq.Program() as program:
            pq.Q() | pq.FockStateVector(
                {
                    (2, 1, 0, 3): 0.3,
                    (1, 1, 2, 3): 0.2,
                    ...
                }
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._simulators.fock.pure.state.PureFockState` and
    :class:`~piquasso._simulators.sampling.state.SamplingState`.
    """

    def __init__(
        self,
        fock_amplitude_map: Dict[Tuple[int, ...], complex],
        coefficient: complex = 1.0,
    ) -> None:
        """
        Args:
            fock_amplitude_map (Dict[Tuple[int, ...], complex]): The Fock amplitude map.
            coefficient (complex, optional):
                The coefficient of the occupation number. Defaults to :math:`1.0`.

        Raises:
            InvalidState:
                If the specified occupation numbers are not all natural numbers.
        """
        if not all(all_natural(occupation) for occupation in fock_amplitude_map.keys()):
            raise InvalidState(
                f"Invalid occupation numbers in "
                f"fock_amplitude_map: {fock_amplitude_map}\n"
                "Occupation numbers must contain non-negative integers."
            )
        super().__init__(
            params=dict(
                fock_amplitude_map=fock_amplitude_map,
                coefficient=coefficient,
            ),
        )

    def __add__(self, other):
        if isinstance(other, NumberState):
            fock_amplitude_map = {
                occupation_numbers: coefficient * self.params["coefficient"]
                for occupation_numbers, coefficient in self.params[
                    "fock_amplitude_map"
                ].items()
            }

            if other.params["occupation_numbers"] in self.params["fock_amplitude_map"]:
                fock_amplitude_map[other.params["occupation_numbers"]] += other.params[
                    "coefficient"
                ]
            else:
                fock_amplitude_map[other.params["occupation_numbers"]] = other.params[
                    "coefficient"
                ]

            return FockStateVector(fock_amplitude_map=fock_amplitude_map)

        elif isinstance(other, FockStateVector):
            fock_amplitude_map = {
                occupation_numbers: coefficient * self.params["coefficient"]
                for occupation_numbers, coefficient in self.params[
                    "fock_amplitude_map"
                ].items()
            }
            for occupation_numbers, coefficient in other.params[
                "fock_amplitude_map"
            ].items():
                coefficient *= other.params["coefficient"]
                if occupation_numbers in fock_amplitude_map:
                    fock_amplitude_map[occupation_numbers] += coefficient
                else:
                    fock_amplitude_map[occupation_numbers] = coefficient

            return FockStateVector(fock_amplitude_map=fock_amplitude_map)
        else:
            return NotImplemented


class StateVector(Preparation, _mixins.WeightMixin):
    r"""State preparation with Fock basis vectors.

    Example usage::

        with pq.Program() as program:
            pq.Q() | pq.StateVector([2, 1, 0, 3])
            ...

    Can only be applied to the following states:
    :class:`~piquasso._simulators.fock.pure.state.PureFockState`.

    .. deprecated:: 8.0.0
       Use :class:`NumberState` or :class:`FockStateVector` instead.
    """

    def __init__(
        self,
        occupation_numbers: Optional[Iterable[int]] = None,
        fock_amplitude_map: Optional[Dict[Tuple[int, ...], complex]] = None,
        coefficient: complex = 1.0,
    ) -> None:
        """
        Args:
            occupation_numbers (Iterable[int], optional): The occupation numbers.
            fock_amplitude_map (Dict[Tuple[int, ...], complex], optional):
                A mapping of occupation numbers to their corresponding amplitudes.
            coefficient (complex, optional):
                The coefficient of the occupation number. Defaults to :math:`1.0`.

        Raises:
            InvalidParameter:
                If neither `occupation_numbers` nor `fock_amplitude_map` is provided.
            InvalidParameter:
                If both `occupation_numbers` and `fock_amplitude_map` are provided.
            InvalidState:
                If the specified occupation numbers are not all natural numbers.
            InvalidState:
                If the keys in `fock_amplitude_map` are not all natural numbers.
        """

        warnings.warn(
            "`pq.StateVector` is deprecated and will be removed in a future "
            "release. Use `pq.NumberState` or `pq.FockStateVector` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if occupation_numbers is None and fock_amplitude_map is None:
            raise InvalidParameter(
                "Either 'occupation_numbers' or 'fock_amplitude_map' must be provided."
            )

        if occupation_numbers is not None and fock_amplitude_map is not None:
            raise InvalidParameter(
                "Only one of 'occupation_numbers' or 'fock_amplitude_map' "
                "can be provided."
            )

        if occupation_numbers is not None:
            if not all_natural(occupation_numbers):
                raise InvalidState(
                    f"Invalid occupation numbers: "
                    f"occupation_numbers={occupation_numbers}\n"
                    "Occupation numbers must contain non-negative integers."
                )
            super().__init__(
                params=dict(
                    occupation_numbers=tuple(occupation_numbers),
                    coefficient=coefficient,
                ),
            )

        if fock_amplitude_map is not None:
            if not all(
                all_natural(occupation) for occupation in fock_amplitude_map.keys()
            ):
                raise InvalidState(
                    f"Invalid occupation numbers in "
                    f"fock_amplitude_map: {fock_amplitude_map}\n"
                    "Occupation numbers must contain non-negative integers."
                )
            super().__init__(
                params=dict(
                    fock_amplitude_map=fock_amplitude_map,
                    coefficient=coefficient,
                ),
            )


class DensityMatrix(Preparation, _mixins.WeightMixin):
    r"""State preparation with density matrix elements.

    Example usage::

        with pq.Program() as program:
            pq.Q() | (
                0.2 * pq.DensityMatrix(ket=(1, 0, 1), bra=(2, 1, 0))
                + 0.3 * pq.DensityMatrix(ket=(2, 0, 1), bra=(0, 1, 1))
                ...
            )
            ...

    Note:
        This only creates one matrix element.

    Can only be applied to the following states:
    :class:`~piquasso._simulators.fock.general.state.FockState`.
    """

    def __init__(
        self,
        ket: Iterable[int],
        bra: Iterable[int],
        coefficient: complex = 1.0,
    ) -> None:
        """
        Args:
            bra (Iterable[int]): The bra vector.
            ket (Iterable[int]): The ket vector.
            coefficient (complex, optional):
                The coefficient of the operator defined by the "bra" and "ket" vectors.
                Defaults to :math:`1.0`.

        Raises:
            InvalidState:
                If the specified "bra" or "ket" vectors are not all natural numbers.
        """

        if not all_natural(ket):
            raise InvalidState(f"Invalid ket vector: ket={ket}")

        if not all_natural(bra):
            raise InvalidState(f"Invalid ket vector: ket={bra}")

        super().__init__(
            params=dict(
                ket=tuple(ket),
                bra=tuple(bra),
                coefficient=coefficient,
            ),
        )


class Create(Preparation):
    r"""Create a particle on a mode.

    This instruction essentially applies a creation operator on the specified mode, then
    normalizes the state.

    Example usage::

        with pq.Program() as program:
            pq.Q(1) | pq.Create()
            pq.Q(2) | pq.Create()
            ...

    Can only be applied to the following states:
    :class:`~piquasso._simulators.fock.general.state.FockState`,
    :class:`~piquasso._simulators.fock.pure.state.PureFockState`.
    """

    def __init__(self) -> None:
        super().__init__()


class Annihilate(Preparation):
    r"""Annihilate a particle on a mode.

    This instruction essentially applies an annihilation operator on the specified mode,
    then normalizes the state.

    Example usage::

        with pq.Program() as program:
            ...
            pq.Q(0) | pq.Annihilate()
            ...

    Can only be applied to the following states:
    :class:`~piquasso._simulators.fock.general.state.FockState`,
    :class:`~piquasso._simulators.fock.pure.state.PureFockState`.
    """

    def __init__(self) -> None:
        super().__init__()
