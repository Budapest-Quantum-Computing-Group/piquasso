#
# Copyright (C) 2020 by TODO - All rights reserved.
#

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
