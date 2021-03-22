#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import random
import numpy as np

from ..state import BaseFockState

from .circuit import PNCFockCircuit

from scipy.linalg import block_diag


class PNCFockState(BaseFockState):
    circuit_class = PNCFockCircuit

    def __init__(self, representation=None, *, d, cutoff):
        super().__init__(d=d, cutoff=cutoff)

        self._representation = (
            np.array(representation)
            if representation is not None
            else self._get_empty()
        )

    @classmethod
    def from_number_preparations(cls, *, d, cutoff, number_preparations):
        """
        TODO: Remove coupling!
        """

        self = cls(d=d, cutoff=cutoff)

        for number_preparation in number_preparations:
            ket, bra, coefficient = number_preparation.params

            self._add_occupation_number_basis(
                ket=ket,
                bra=bra,
                coefficient=coefficient,
            )

        return self

    def _get_empty(self):
        return [
            np.zeros(shape=(self._space._symmetric_cardinality(n), ) * 2, dtype=complex)
            for n in range(self._space.cutoff + 1)
        ]

    def _apply_vacuum(self):
        self._representation = self._get_empty()
        self._representation[0][0, 0] = 1.0

    def _apply_passive_linear(self, operator, modes):
        index = self._get_operator_index(modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        for n, subrep in enumerate(self._representation):
            tensorpower_operator = self._space.symmetric_tensorpower(
                embedded_operator, n
            )
            self._representation[n] = (
                tensorpower_operator @ subrep
                @ tensorpower_operator.conjugate().transpose()
            )

    def _simulate_collapse_on_modes(self, *, modes):
        probability_map = {}

        for n, subrep in enumerate(self._representation):
            for index, basis in self._space.subspace_operator_basis_diagonal_on_modes(
                modes=modes, n=n
            ):
                coefficient = self._representation[n][index]

                subspace_basis = basis.ket.on_modes(modes=modes)

                if subspace_basis in probability_map:
                    probability_map[subspace_basis] += coefficient
                else:
                    probability_map[subspace_basis] = coefficient

        outcome = random.choices(
            population=list(probability_map.keys()),
            weights=probability_map.values(),
        )[0]

        normalization = 1 / probability_map[outcome].real

        return outcome, normalization

    def _project_to_subspace(self, *, subspace_basis, modes, normalization):
        projected_representation = self._get_projected(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        for n, subrep in enumerate(projected_representation):
            self._representation[n] = subrep * normalization

    def _get_projected(self, *, subspace_basis, modes):
        new_representation = self._get_empty()

        for n, subrep in enumerate(self._representation):
            index = self._space.get_projection_operator_indices_on_subspace(
                subspace_basis=subspace_basis,
                modes=modes,
                n=n,
            )

            if index:
                new_representation[n][index] = subrep[index]

        return new_representation

    def _hacky_apply_operator(self, operator):
        """
        HACK: Here we switch to the full representation for a brief moment. I'm sure
        there's a better way.
        """

        density_matrix = block_diag(*self._representation)

        density_matrix = operator @ density_matrix @ operator.transpose()

        for n, subrep in enumerate(self._representation):
            begin, end = self._space.get_subspace_indices(n)

            self._representation[n] = density_matrix[begin:end, begin:end]

    def _apply_creation_operator(self, modes):
        operator = self._space.get_creation_operator(modes)

        self._hacky_apply_operator(operator)

        self.normalize()

    def _apply_annihilation_operator(self, modes):
        operator = self._space.get_annihilation_operator(modes)

        self._hacky_apply_operator(operator)

        self.normalize()

    def _add_occupation_number_basis(self, *, ket, bra, coefficient):
        index = self._space.index(ket)
        dual_index = self._space.index(bra)

        ket_n = sum(ket)
        bra_n = sum(bra)

        if ket_n != bra_n:
            return

        n = ket_n

        subspace_basis = self._space.get_subspace_operator_basis(n)

        index = subspace_basis.index(ket)
        dual_index = subspace_basis.index(bra)

        self._representation[n][index, dual_index] = coefficient

    def _apply_kerr(self, xi, mode):
        for n, subrep in enumerate(self._representation):
            for index, (basis, dual_basis) in (
                self._space.enumerate_subspace_operator_basis(n)
            ):
                number = basis[mode]
                dual_number = dual_basis[mode]

                coefficient = np.exp(
                    1j * xi * (
                        number * (2 * number + 1)
                        - dual_number * (2 * dual_number + 1)
                    )
                )

                self._representation[n][index] *= coefficient

    def _apply_cross_kerr(self, xi, modes):
        for n, subrep in enumerate(self._representation):
            for index, (basis, dual_basis) in (
                self._space.enumerate_subspace_operator_basis(n)
            ):
                coefficient = np.exp(
                    1j * xi * (
                        basis[modes[0]] * basis[modes[1]]
                        - dual_basis[modes[0]] * dual_basis[modes[1]]
                    )
                )

                self._representation[n][index] *= coefficient

    @property
    def nonzero_elements(self):
        for n, subrep in enumerate(self._representation):
            for index, basis in self._space.enumerate_subspace_operator_basis(n):
                coefficient = self._representation[n][index]
                if coefficient != 0:
                    yield coefficient, basis

    def __repr__(self):
        return " + ".join(
            [
                str(coefficient) + str(basis)
                for coefficient, basis in self.nonzero_elements
            ]
        )

    def __eq__(self, other):
        return all([
            np.allclose(subrep, other._representation[n])
            for n, subrep in enumerate(self._representation)
        ])

    @property
    def fock_probabilities(self):
        ret = []

        for subrep in self._representation:
            ret.extend(np.diag(subrep))

        return ret

    def normalize(self):
        if np.isclose(self.norm, 0):
            raise RuntimeError("The norm of the state is 0.")

        norm = self.norm

        for n, subrep in enumerate(self._representation):
            self._representation[n] = subrep / norm