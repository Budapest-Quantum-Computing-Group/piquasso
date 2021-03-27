#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from ..state import BaseFockState

from .circuit import FockCircuit


class FockState(BaseFockState):
    circuit_class = FockCircuit

    def __init__(self, density_matrix=None, *, d, cutoff):
        super().__init__(d=d, cutoff=cutoff)

        self._density_matrix = (
            np.array(density_matrix)
            if density_matrix is not None
            else self._get_empty()
        )

    @classmethod
    def from_pure(cls, pure_fock_state):
        state_vector = pure_fock_state._state_vector
        density_matrix = np.outer(state_vector, state_vector)

        return cls(
            density_matrix=density_matrix,
            d=pure_fock_state.d,
            cutoff=pure_fock_state.cutoff,
        )

    def _get_empty(self):
        return np.zeros(shape=(self._space.cardinality, ) * 2, dtype=complex)

    def _apply_vacuum(self):
        self._density_matrix = self._get_empty()
        self._density_matrix[0, 0] = 1.0

    def _apply_passive_linear(self, operator, modes):
        index = self._get_operator_index(modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        fock_operator = self._space.get_fock_operator(embedded_operator)

        self._density_matrix = (
            fock_operator @ self._density_matrix @ fock_operator.conjugate().transpose()
        )

    def _get_probability_map(self, *, modes):
        probability_map = {}

        for index, basis in self._space.operator_basis_diagonal_on_modes(modes=modes):
            coefficient = self._density_matrix[index]

            subspace_basis = basis.ket.on_modes(modes=modes)

            if subspace_basis in probability_map:
                probability_map[subspace_basis] += coefficient
            else:
                probability_map[subspace_basis] = coefficient

        return probability_map

    @staticmethod
    def _get_normalization(probability_map, sample):
        return 1 / probability_map[sample].real

    def _project_to_subspace(self, *, subspace_basis, modes, normalization):
        projected_density_matrix = self._get_projected_density_matrix(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        self._density_matrix = projected_density_matrix * normalization

    def _get_projected_density_matrix(self, *, subspace_basis, modes):
        new_density_matrix = self._get_empty()

        index = self._space.get_projection_operator_indices(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        new_density_matrix[index] = self._density_matrix[index]

        return new_density_matrix

    def _apply_creation_operator(self, modes):
        operator = self._space.get_creation_operator(modes)

        self._density_matrix = operator @ self._density_matrix @ operator.transpose()

        self.normalize()

    def _apply_annihilation_operator(self, modes):
        operator = self._space.get_annihilation_operator(modes)

        self._density_matrix = operator @ self._density_matrix @ operator.transpose()

        self.normalize()

    def _add_occupation_number_basis(self, *, ket, bra, coefficient):
        index = self._space.index(ket)
        dual_index = self._space.index(bra)

        self._density_matrix[index, dual_index] = coefficient

    def _apply_kerr(self, xi, mode):
        for index, (basis, dual_basis) in self._space.operator_basis:
            number = basis[mode]
            dual_number = dual_basis[mode]

            coefficient = np.exp(
                1j * xi * (
                   number * (2 * number + 1)
                   - dual_number * (2 * dual_number + 1)
                )
            )

            self._density_matrix[index] *= coefficient

    def _apply_cross_kerr(self, xi, modes):
        for index, (basis, dual_basis) in self._space.operator_basis:
            coefficient = np.exp(
                1j * xi * (
                    basis[modes[0]] * basis[modes[1]]
                    - dual_basis[modes[0]] * dual_basis[modes[1]]
                )
            )

            self._density_matrix[index] *= coefficient

    @property
    def nonzero_elements(self):
        for index, basis in self._space.operator_basis:
            coefficient = self._density_matrix[index]
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
        return np.allclose(self._density_matrix, other._density_matrix)

    @property
    def fock_probabilities(self):
        return np.diag(self._density_matrix).real

    def normalize(self):
        if np.isclose(self.norm, 0):
            raise RuntimeError("The norm of the state is 0.")

        self._density_matrix = self._density_matrix / self.norm
