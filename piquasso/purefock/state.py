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
            state_vector = np.zeros(shape=(space.cardinality, ))

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
        basis_vectors = self._space.basis_vectors

        probabilities = self.fock_probabilities

        index = random.choices(range(len(basis_vectors)), probabilities)[0]

        outcome_basis_vector = basis_vectors[index]

        outcome = tuple(outcome_basis_vector)

        new_state_vector = np.zeros(
            shape=self._state_vector.shape,
            dtype=complex,
        )

        new_state_vector[index] = self._state_vector[index]

        self._state_vector = new_state_vector / np.sqrt(probabilities[index])

        return outcome

    def _add_occupation_number_basis(self, coefficient, occupation_numbers):
        index = self._space.get_index_by_occupation_basis(occupation_numbers)

        self._state_vector[index] = coefficient

    def __repr__(self):
        basis_vectors = self._space.basis_vectors

        ret = []

        for (index, ket) in enumerate(basis_vectors):
            vector_element = self._state_vector[index]
            if vector_element == 0:
                continue

            ret.append(
                str(vector_element)
                + str(ket)
            )

        return " + ".join(ret)

    @property
    def fock_probabilities(self):
        return self._state_vector * self._state_vector.conjugate()
