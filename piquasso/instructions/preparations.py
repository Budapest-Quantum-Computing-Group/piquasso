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

from typing import Tuple

import numpy as np

from piquasso.core import _mixins
from piquasso.api.instruction import Preparation


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
        pass


class Mean(Preparation):
    r"""Set the first canonical moment of the state.

    Example usage::

        with pq.Program() as program:
            pq.Q() | pq.Mean(
                mean=np.array(...)
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._backends.gaussian.state.GaussianState`.
    """

    def __init__(self, mean: np.ndarray) -> None:
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
    :class:`~piquasso._backends.gaussian.state.GaussianState`.
    """

    def __init__(self, cov: np.ndarray) -> None:
        super().__init__(params=dict(cov=cov))


class StateVector(Preparation, _mixins.WeightMixin):
    r"""State preparation with Fock basis vectors.

    Example usage::

        with pq.Program() as program:
            pq.Q() | (
                0.3 * pq.StateVector(2, 1, 0, 3)
                + 0.2 * pq.StateVector(1, 1, 2, 3)
                ...
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._backends.fock.pure.state.PureFockState`.
    """

    def __init__(
        self, *occupation_numbers: complex, coefficient: complex = 1.0
    ) -> None:
        super().__init__(
            params=dict(
                occupation_numbers=occupation_numbers,
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
    :class:`~piquasso._backends.fock.general.state.FockState`.
    """

    def __init__(
        self,
        ket: Tuple[complex, ...] = None,
        bra: Tuple[complex, ...] = None,
        coefficient: complex = 1.0,
    ) -> None:
        super().__init__(
            params=dict(
                ket=ket,
                bra=bra,
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
    :class:`~piquasso._backends.fock.general.state.FockState`,
    :class:`~piquasso._backends.fock.pure.state.PureFockState`.
    """

    def __init__(self) -> None:
        pass


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
    :class:`~piquasso._backends.fock.general.state.FockState`,
    :class:`~piquasso._backends.fock.pure.state.PureFockState`.
    """

    def __init__(self) -> None:
        pass
