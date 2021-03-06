#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import random
import numpy as np

from piquasso.api.state import State

from piquasso._math import fock

from .circuit import PureFockCircuit


class PureFockState(State):
    _circuit_class = PureFockCircuit

    def __init__(self, state_vector=None, *, d, cutoff, vacuum=False):
        space = fock.FockSpace(
            d=d,
            cutoff=cutoff,
        )

        if state_vector is None:
            state_vector = np.zeros(shape=(space.cardinality, ), dtype=complex)

            if vacuum is True:
                state_vector[0] = 1.0

        self._state_vector = np.array(state_vector)
        self._space = space

    @classmethod
    def create_vacuum(cls, *, d, cutoff):
        return cls(d=d, cutoff=cutoff, vacuum=True)

    def _apply(self, operator, modes):
        index = self._get_operator_index(modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        fock_operator = self._space.get_fock_operator(embedded_operator)

        self._state_vector = fock_operator @ self._state_vector

    def _measure_particle_number(self):
        probabilities = self.fock_probabilities

        index = random.choices(range(len(self._space)), probabilities)[0]

        outcome_basis_vector = self._space[index]

        outcome = tuple(outcome_basis_vector)

        new_state_vector = np.zeros(
            shape=self._state_vector.shape,
            dtype=complex,
        )

        new_state_vector[index] = self._state_vector[index]

        self._state_vector = new_state_vector / np.sqrt(probabilities[index])

        return outcome

    def _add_occupation_number_basis(self, coefficient, occupation_numbers):
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
