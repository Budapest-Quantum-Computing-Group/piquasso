#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import random
import numpy as np

from ..state import BaseFockState

from .circuit import PureFockCircuit


class PureFockState(BaseFockState):
    circuit_class = PureFockCircuit

    def __init__(self, state_vector=None, *, d, cutoff, vacuum=False):
        super().__init__(d=d, cutoff=cutoff)

        if state_vector is None:
            state_vector = self._get_empty_state_vector(
                cardinality=self._space.cardinality
            )

            if vacuum is True:
                state_vector[0] = 1.0

        self._state_vector = np.array(state_vector)

    @classmethod
    def create_vacuum(cls, *, d, cutoff):
        return cls(d=d, cutoff=cutoff, vacuum=True)

    @classmethod
    def from_number_preparations(cls, *, d, cutoff, number_preparations):
        """
        TODO: Remove coupling!
        """

        self = cls(d=d, cutoff=cutoff)

        for number_preparation in number_preparations:
            self._add_occupation_number_basis(
                occupation_numbers=number_preparation.params[0],
                coefficient=number_preparation.params[-1],
                modes=None,
            )

        return self

    def _get_empty_state_vector(self, cardinality):
        return np.zeros(shape=(cardinality, ), dtype=complex)

    def _apply_passive_linear(self, operator, modes):
        index = self._get_operator_index(modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        fock_operator = self._space.get_fock_operator(embedded_operator)

        self._state_vector = fock_operator @ self._state_vector

    def _measure_particle_number(self, modes):
        if not modes:
            modes = tuple(range(self._space.d))

        outcome, probability = self._simulate_collapse_on_modes(modes=modes)

        self._project_to_subspace(
            subspace_basis=outcome,
            modes=modes,
            normalization=np.sqrt(1 / probability),
        )

        return outcome

    def _simulate_collapse_on_modes(self, *, modes):
        probability_map = {}

        for index, basis in self._space.basis:
            coefficient = self._state_vector[index]

            if np.isclose(coefficient, 0):
                # TODO: Do we need this?
                continue

            subspace_basis = basis.on_modes(modes=modes)

            if subspace_basis in probability_map:
                probability_map[subspace_basis] += coefficient ** 2
            else:
                probability_map[subspace_basis] = coefficient ** 2

        outcome = random.choices(
            population=list(probability_map.keys()),
            weights=probability_map.values(),
        )[0]

        return outcome, probability_map[outcome]

    def _project_to_subspace(self, *, subspace_basis, modes, normalization):
        projected_state_vector = self._get_projected_state_vector(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        self._state_vector = projected_state_vector * normalization

    def _get_projected_state_vector(self, *, subspace_basis, modes):
        new_state_vector = self._get_empty_state_vector(
            cardinality=self._space.cardinality
        )

        index = self._space.get_projection_operator_indices_for_pure(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        new_state_vector[index] = self._state_vector[index]

        return new_state_vector

    def _add_occupation_number_basis(self, coefficient, occupation_numbers, modes):
        if modes:
            occupation_numbers = self._space.get_occupied_basis(
                modes=modes, occupation_numbers=occupation_numbers
            )

        index = self._space.index(occupation_numbers)

        self._state_vector[index] = coefficient

    def _apply_creation_operator(self, modes):
        operator = self._space.get_creation_operator(modes)

        self._state_vector = operator @ self._state_vector

        self.normalize()

    def _apply_annihilation_operator(self, modes):
        operator = self._space.get_annihilation_operator(modes)

        self._state_vector = operator @ self._state_vector

        self.normalize()

    def _apply_kerr(self, xi, mode):
        for index, basis in self._space.basis:
            number = basis[mode]
            coefficient = np.exp(
                1j * xi * number * (2 * number + 1)
            )
            self._state_vector[index] *= coefficient

    def _apply_cross_kerr(self, xi, modes):
        for index, basis in self._space.basis:
            coefficient = np.exp(
                1j * xi * basis[modes[0]] * basis[modes[1]]
            )
            self._state_vector[index] *= coefficient

    @property
    def nonzero_elements(self):
        for index, basis in self._space.basis:
            coefficient = self._state_vector[index]
            if coefficient != 0:
                yield coefficient, basis

    def __repr__(self):
        return " + ".join([
            str(coefficient) + str(basis)
            for coefficient, basis in self.nonzero_elements
        ])

    def __eq__(self, other):
        return np.allclose(self._state_vector, other._state_vector)

    @property
    def fock_probabilities(self):
        return (self._state_vector * self._state_vector.conjugate()).real

    @property
    def norm(self):
        return sum(self.fock_probabilities)

    def normalize(self):
        if np.isclose(self.norm, 0):
            raise RuntimeError("The norm of the state is 0.")

        self._state_vector = self._state_vector / np.sqrt(self.norm)
