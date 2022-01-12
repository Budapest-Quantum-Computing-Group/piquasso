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

import functools
from typing import Tuple, Iterable, Generator, Any, List

import numpy as np

from operator import add

from scipy.special import factorial, comb
from scipy.linalg import block_diag, polar, logm

from piquasso._math.indices import get_operator_index
from piquasso._math.combinatorics import partitions
from piquasso._math.decompositions import takagi

from piquasso.api.typing import PermanentFunction


@functools.lru_cache()
def cutoff_cardinality(*, cutoff: int, d: int) -> int:
    r"""
    Calculates the dimension of the cutoff Fock space with the relation

    ..math::
        \sum_{i=0}^{c - 1} {d + i - 1 \choose i} = {d + c - 1 \choose c - 1}.
    """
    return comb(d + cutoff - 1, cutoff - 1, exact=True)


@functools.lru_cache()
def symmetric_subspace_cardinality(*, d: int, n: int) -> int:
    return comb(d + n - 1, n, exact=True)


class FockBasis(tuple):
    def __str__(self) -> str:
        return self.display(template="|{}>")

    def display(self, template: str = "{}") -> str:
        return template.format("".join([str(number) for number in self]))

    def display_as_bra(self) -> str:
        return self.display("<{}|")

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, other: Iterable) -> "FockBasis":
        return FockBasis(map(add, self, other))

    __radd__ = __add__

    @property
    def d(self) -> int:
        return len(self)

    @property
    def n(self) -> int:
        return sum(self)

    @classmethod
    def create_on_particle_subspace(
        cls, *, boxes: int, particles: int
    ) -> List[Tuple[int, ...]]:
        return partitions(boxes=boxes, particles=particles, class_=cls)

    @classmethod
    def create_all(cls, *, d: int, cutoff: int) -> List[Tuple[int, ...]]:
        ret = []

        for n in range(cutoff):
            ret.extend(cls.create_on_particle_subspace(boxes=d, particles=n))

        return ret

    def on_modes(self, *, modes: Tuple[int, ...]) -> "FockBasis":
        return FockBasis(self[mode] for mode in modes)

    def increment_on_modes(self, modes: Tuple[int, ...]) -> "FockBasis":
        a = [0] * self.d
        for mode in modes:
            a[mode] = 1

        return self + a


class FockOperatorBasis(tuple):
    def __new__(cls, *, ket: Iterable, bra: Iterable) -> "FockOperatorBasis":
        return super().__new__(cls, (FockBasis(ket), FockBasis(bra)))  # type: ignore

    def __str__(self) -> str:
        return str(self.ket) + self.bra.display_as_bra()

    @property
    def ket(self) -> FockBasis:
        return self[0]

    @property
    def bra(self) -> FockBasis:
        return self[1]

    def is_diagonal_on_modes(self, modes: Tuple[int, ...]) -> bool:
        return self.ket.on_modes(modes=modes) == self.bra.on_modes(modes=modes)


class FockSpace(tuple):
    r"""
    Note, that when you symmetrize a tensor, i.e. use the superoperator

    .. math::
        S: A \otimes B \mapsto A \vee B

    on a tensor which is expressed in the regular Fock basis, then the resulting tensor
    still remains in the regular representation. You have to perform a basis
    transformation to acquire the symmetrized tensor in the symmetrized representation.
    """

    def __new__(cls, d: int, cutoff: int) -> "FockSpace":
        return super().__new__(
            cls, FockBasis.create_all(d=d, cutoff=cutoff)  # type: ignore
        )

    def __init__(self, *, d: int, cutoff: int) -> None:
        self.d = d
        self.cutoff = cutoff

    def __deepcopy__(self, memo: Any) -> "FockSpace":
        """
        This method exists, because `copy.deepcopy` produces errors with classes
        defining both `__new__` and `__init__`.

        It defines the deepcopy of this object. Since its state (:attr:`d` and
        :attr:`cutoff`) is immutable, we don't really need to deepcopy this object, we
        could return with this instance, too.
        """

        return self

    def get_passive_fock_operator(
        self,
        operator: np.ndarray,
        modes: Tuple[int, ...],
        d: int,
        permanent_function: PermanentFunction,
    ) -> np.ndarray:
        index = get_operator_index(modes)

        embedded_operator = np.identity(d, dtype=complex)

        embedded_operator[index] = operator

        return block_diag(
            *(
                self.symmetric_tensorpower(
                    embedded_operator, n, permanent_function=permanent_function
                )
                for n in range(self.cutoff)
            )
        )

    def get_single_mode_displacement_operator(
        self,
        *,
        r: float,
        phi: float,
    ) -> np.ndarray:

        """
        This method generates the Displacement operator following a recursion rule.
        Reference: https://quantum-journal.org/papers/q-2020-11-30-366/.

        Args:
        r (float): This is the Displacement amplitude. Typically this value can be
            negative or positive depending on the desired displacement direction.
            Note:
                Setting :math:`|r|` to higher values will require you to have a higer
                cuttof dimensions.
        phi (float): This is the Dispalacement angle. Its ranges are
            :math:`\phi \in [ 0, 2 \pi )`

        Returns:
            np.ndarray: The constructed Displacement matrix representing the Fock
            operator.
        """

        fock_indices = np.sqrt(np.arange(self.cutoff, dtype=complex))
        displacement = r * np.exp(1j * phi)

        transformation = np.zeros((self.cutoff,) * 2, dtype=complex)
        transformation[0, 0] = np.exp(-0.5 * r ** 2)

        for row in range(1, self.cutoff):
            transformation[row, 0] = (
                displacement / fock_indices[row] * transformation[row - 1, 0]
            )

        for row in range(self.cutoff):
            for col in range(1, self.cutoff):
                transformation[row, col] = (
                    -displacement.conj()
                    / fock_indices[col]
                    * transformation[row, col - 1]
                ) + (
                    fock_indices[row]
                    / fock_indices[col]
                    * transformation[row - 1, col - 1]
                )

        return transformation

    def get_single_mode_squeezing_operator(
        self,
        *,
        r: float,
        phi: float,
    ) -> np.ndarray:
        """
        This method generates the Squeezing operator following a recursion rule.
        Reference: https://quantum-journal.org/papers/q-2020-11-30-366/.

        Args:
        r (float): This is the Squeezing amplitude. Typically this value can be
            negative or positive depending on the desired squeezing direction.
            Note:
                Setting :math:`|r|` to higher values will require you to have a higer
                cuttof dimensions.
        phi (float): This is the Squeezing angle. Its ranges are
            :math:`\phi \in [ 0, 2 \pi )`

        Returns:
            np.ndarray: The constructed Squeezing matrix representing the Fock operator.
        """

        sechr = 1.0 / np.cosh(r)
        A = np.exp(1j * phi) * np.tanh(r)

        transformation = np.zeros((self.cutoff,) * 2, dtype=complex)
        transformation[0, 0] = np.sqrt(sechr)

        fock_indices = np.sqrt(np.arange(self.cutoff, dtype=complex))

        for index in range(2, self.cutoff, 2):
            transformation[index, 0] = (
                -fock_indices[index - 1]
                / fock_indices[index]
                * (transformation[index - 2, 0] * A)
            )

        for row in range(0, self.cutoff):
            for col in range(1, self.cutoff):
                if (row + col) % 2 == 0:
                    transformation[row, col] = (
                        1
                        / fock_indices[col]
                        * (
                            (
                                fock_indices[row]
                                * transformation[row - 1, col - 1]
                                * sechr
                            )
                            + (
                                fock_indices[col - 1]
                                * A.conj()
                                * transformation[row, col - 2]
                            )
                        )
                    )

        return transformation

    def embed_matrix(
        self,
        matrix: np.ndarray,
        modes: Tuple[int, ...],
        auxiliary_modes: Tuple[int, ...],
    ) -> np.ndarray:
        embedded_matrix = np.zeros((self.cardinality,) * 2, dtype=complex)

        for embedded_index, operator_basis in self.operator_basis_diagonal_on_modes(
            modes=auxiliary_modes
        ):
            index = (
                operator_basis.ket.on_modes(modes=modes),
                operator_basis.bra.on_modes(modes=modes),
            )
            embedded_matrix[embedded_index] = matrix[index]

        return embedded_matrix

    def get_linear_fock_operator(
        self,
        *,
        modes: Tuple[int, ...],
        active_block: np.ndarray,
        passive_block: np.ndarray,
        permanent_function: PermanentFunction,
    ) -> np.ndarray:
        r"""The matrix of the symplectic transformation in Fock space.

        Any symplectic transformation (in complex representation) can be written as

        .. math::
            S = \begin{bmatrix}
                P & A \\
                A^* & P^*
            \end{bmatrix}.

        As a first step, this symplectic matrix is polar decomposed:

        .. math::
            S = R U,

        where :math:`R` is a hermitian matrix and :math:`U` is a unitary one. The polar
        decomposition of a symplectic matrix is also a symplectic matrix, therefore
        :math:`U` (being unitary) corresponds to a passive transformation, and :math:`R`
        to an active one.

        The symplectic matrix :math:`R` has the form

        .. math::
            \exp \left ( i K \begin{bmatrix}
                0 & Z \\
                Z^* & 0
            \end{bmatrix}
            \right ),

        where :math:`Z` is a (complex) symmetric matrix. This can be decomposed via
        Takagi decomposition:

        .. math::
            Z = U D U^T,

        where :math:`U` is a unitary matrix and :math:`D` is a diagonal matrix. The
        diagonal entries in :math:`D` correspond to squeezing amplitudes, and :math:`U`
        corresponds to an interferometer.

        Args:
            modes (Tuple[int, ...]):
                The modes on which the transformation should be applied.
            active_block (np.ndarray): Active part of the symplectic transformation.
            passive_block (np.ndarray): Passive part of the symplectic transformation.

        Returns:
            np.ndarray:
                The resulting transformation, which could be applied to the state.
        """

        d = len(modes)
        identity = np.identity(d)
        zeros = np.zeros_like(identity)

        K = np.block(
            [
                [identity, zeros],
                [zeros, -identity],
            ],
        )

        symplectic = np.block(
            [
                [passive_block, active_block],
                [active_block.conj(), passive_block.conj()],
            ],
        )

        U, R = polar(symplectic, side="left")

        H_active = 1j * K @ logm(R)
        H_passive = 1j * K @ logm(U)

        singular_values, unitary = takagi(1j * H_active[:d, d:])

        transformation = np.identity(self.cardinality, dtype=complex)

        fock_operator = self.get_passive_fock_operator(
            unitary.conj().T @ H_passive[:d, :d],
            modes=modes,
            d=self.d,
            permanent_function=permanent_function,
        )

        transformation = fock_operator @ transformation

        for index, mode in enumerate(modes):
            operator = self.get_single_mode_squeezing_operator(
                r=singular_values[index],
                phi=0.0,
            )

            squeezing_matrix = self.embed_matrix(
                operator,
                modes=(mode,),
                auxiliary_modes=tuple(np.delete(np.arange(self.d), (mode,))),
            )

            transformation = squeezing_matrix @ transformation

        fock_operator = self.get_passive_fock_operator(
            unitary,
            modes=modes,
            d=self.d,
            permanent_function=permanent_function,
        )

        transformation = fock_operator @ transformation

        return transformation

    @property
    def cardinality(self) -> int:
        return cutoff_cardinality(cutoff=self.cutoff, d=self.d)

    @property
    def basis(self) -> Generator[Tuple[int, FockBasis], Any, None]:
        yield from enumerate(self)

    @property
    def operator_basis(
        self,
    ) -> Generator[Tuple[Tuple[int, int], FockOperatorBasis], Any, None]:
        for index, basis in self.basis:
            for dual_index, dual_basis in self.basis:
                yield (index, dual_index), FockOperatorBasis(ket=basis, bra=dual_basis)

    def operator_basis_diagonal_on_modes(
        self, *, modes: Tuple[int, ...]
    ) -> Generator[Tuple[Tuple[int, int], FockOperatorBasis], Any, None]:
        yield from [
            (index, basis)
            for index, basis in self.operator_basis
            if basis.is_diagonal_on_modes(modes=modes)
        ]

    def get_occupied_basis(
        self, *, modes: Tuple[int, ...], occupation_numbers: Tuple[int, ...]
    ) -> FockBasis:
        temp = [0] * self.d
        for index, mode in enumerate(modes):
            temp[mode] = occupation_numbers[index]

        return FockBasis(temp)

    def get_projection_operator_indices_for_pure(
        self, *, subspace_basis: Tuple[int, ...], modes: Tuple[int, ...]
    ) -> List[int]:
        return [
            index
            for index, basis in self.basis
            if subspace_basis == basis.on_modes(modes=modes)
        ]

    def get_projection_operator_indices(
        self, *, subspace_basis: Tuple[int, ...], modes: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return tuple(  # type: ignore
            zip(
                *[
                    index
                    for index, operator_basis in self.operator_basis
                    if operator_basis.is_diagonal_on_modes(modes=modes)
                    and subspace_basis == operator_basis.ket.on_modes(modes=modes)
                ]
            )
        )

    def get_subspace_indices(self, n: int) -> Tuple[int, int]:
        begin = cutoff_cardinality(cutoff=n, d=self.d)
        end = cutoff_cardinality(cutoff=n + 1, d=self.d)

        return begin, end

    def get_subspace_basis(self, n: int, d: int = None) -> List[Tuple[int, ...]]:
        d = d or self.d
        return FockBasis.create_on_particle_subspace(boxes=d, particles=n)

    def enumerate_subspace_operator_basis(
        self, n: int, d: int = None
    ) -> Generator[Tuple[Tuple[int, int], FockOperatorBasis], Any, None]:
        d = d or self.d
        subspace_operator_basis = self.get_subspace_basis(n, d)

        for index, basis in enumerate(subspace_operator_basis):
            for dual_index, dual_basis in enumerate(subspace_operator_basis):
                yield (index, dual_index), FockOperatorBasis(ket=basis, bra=dual_basis)

    def symmetric_tensorpower(
        self,
        operator: np.ndarray,
        n: int,
        permanent_function: PermanentFunction,
    ) -> np.ndarray:
        if n == 0:
            return np.array([[1]], dtype=complex)

        if n == 1:
            # NOTE: This stuff is really awkward.
            # The operator in the one-particle-subspace is ordered by e.g.
            # |100>, |010>, |001>, but here we need the order
            # |001>, |010>, |100>.
            return operator[::-1, ::-1].transpose()

        d = len(operator)

        ret = np.empty(
            shape=(symmetric_subspace_cardinality(d=d, n=n),) * 2,
            dtype=complex,
        )

        for index, basis in self.enumerate_subspace_operator_basis(n, d):
            permanent = permanent_function(operator, basis.bra, basis.ket)

            ret[index] = permanent / np.sqrt(
                np.prod(factorial(basis.bra)) * np.prod(factorial(basis.ket))
            )

        return ret

    def get_creation_operator(self, modes: Tuple[int, ...]) -> np.ndarray:
        operator = np.zeros(shape=(self.cardinality,) * 2, dtype=complex)

        for index, basis in enumerate(self):
            dual_basis = basis.increment_on_modes(modes)
            try:
                dual_index = self.index(dual_basis)
                operator[dual_index, index] = 1
            except ValueError:
                # TODO: rethink.
                continue

        return operator

    def get_annihilation_operator(self, modes: Tuple[int, ...]) -> np.ndarray:
        return self.get_creation_operator(modes).transpose()
