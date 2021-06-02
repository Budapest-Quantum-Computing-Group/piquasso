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


from piquasso.core import _mixins
from piquasso.api.instruction import Instruction


class Vacuum(Instruction):
    r"""Prepare the system in a vacuum state.

    This is only practical to be used at the beginning of a program, right after
    specifying the state.

    Example usage:

    .. code-block:: python

        with pq.Program() as program:
            pq.Q() | pq.GaussianState(d=4) | pq.Vacuum()
            ...

    Note:
        This operation can only be used for all modes.
    """

    def __init__(self):
        pass


class Mean(Instruction):
    r"""Set the first canonical moment of the state.

    Example usage:

    .. code-block:: python

        with pq.Program() as program:
            pq.Q() | pq.GaussianState(d=4)

            pq.Q() | pq.Mean(
                mean=np.array(...)
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._backends.gaussian.state.GaussianState`.
    """

    def __init__(self, mean):
        super().__init__(params=dict(mean=mean))


class Covariance(Instruction):
    r"""Sets the covariance matrix of the state.

    Example usage:

    .. code-block:: python

        with pq.Program() as program:
            pq.Q() | pq.GaussianState(d=4)

            pq.Q() | pq.Covariance(
                cov=np.array(...)
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._backends.gaussian.state.GaussianState`.
    """

    def __init__(self, cov):
        super().__init__(params=dict(cov=cov))


class StateVector(Instruction, _mixins.WeightMixin):
    r"""State preparation with Fock basis vectors.

    Example usage:

    .. code-block:: python

        with pq.Program() as program:
            pq.Q() | pq.FockState(d=4) | (
                0.3 * pq.StateVector(2, 1, 0, 3)
                + 0.2 * pq.StateVector(1, 1, 2, 3)
                ...
            )
            ...

    Can only be applied to the following states:
    :class:`~piquasso._backends.fock.pure.state.PureFockState`.
    """

    def __init__(self, *occupation_numbers, coefficient=1.0):
        super().__init__(
            params=dict(
                occupation_numbers=occupation_numbers,
                coefficient=coefficient,
            ),
        )


class DensityMatrix(Instruction, _mixins.WeightMixin):
    r"""State preparation with density matrix elements.

    Example usage:

    .. code-block:: python

        with pq.Program() as program:
            pq.Q() | pq.FockState(d=3) | (
                0.2 * pq.DensityMatrix(ket=(1, 0, 1), bra=(2, 1, 0))
                + 0.3 * pq.DensityMatrix(ket=(2, 0, 1), bra=(0, 1, 1))
                ...
            )
            ...

    Note:
        This only creates one matrix element.

    Can only be applied to the following states:
    :class:`~piquasso._backends.fock.general.state.FockState`,
    :class:`~piquasso._backends.fock.pnc.state.PNCFockState`.
    """

    def __init__(self, ket=None, bra=None, coefficient=1.0):
        super().__init__(
            params=dict(
                ket=ket,
                bra=bra,
                coefficient=coefficient,
            ),
        )


class Create(Instruction):
    r"""Create a particle on a mode.

    .. code-block:: python

        with pq.Program() as program:
            pq.Q() | pq.FockState(d=3, cutoff=4)

            pq.Q(1) | pq.Create()
            pq.Q(2) | pq.Create()
            ...

    Can only be applied to the following states:
    :class:`~piquasso._backends.fock.general.state.FockState`,
    :class:`~piquasso._backends.fock.pure.state.PureFockState`,
    :class:`~piquasso._backends.fock.pnc.state.PNCFockState`.
    """

    def __init__(self):
        pass


class Annihilate(Instruction):
    r"""Annihilate a particle on a mode.

    .. code-block:: python

        with pq.Program() as program:
            pq.Q() | pq.FockState(d=3, cutoff=4)
            ...
            pq.Q(0) | pq.Annihilate()
            ...

    Can only be applied to the following states:
    :class:`~piquasso._backends.fock.general.state.FockState`,
    :class:`~piquasso._backends.fock.pure.state.PureFockState`,
    :class:`~piquasso._backends.fock.pnc.state.PNCFockState`.
    """

    def __init__(self):
        pass
