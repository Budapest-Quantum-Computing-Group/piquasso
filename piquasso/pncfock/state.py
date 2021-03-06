#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import random
import numpy as np

from piquasso.api.state import State

from piquasso._math import fock, direct_sum

from .circuit import PNCFockCircuit


class PNCFockState(State):
    _circuit_class = PNCFockCircuit

    def __init__(self, density_matrix=None, *, d, cutoff, vacuum=False):
        self._density_matrix = density_matrix
        space = fock.FockSpace(
            d=d,
            cutoff=cutoff,
        )

        if density_matrix is None:
            density_matrix = np.zeros(
                (fock.FockSpace(d=d, cutoff=cutoff).cardinality, ) * 2
            )

            if vacuum is True:
                density_matrix[0, 0] = 1.0

        self._density_matrix = density_matrix
        self._space = space

    @classmethod
    def create_vacuum(cls, *, d, cutoff):
        return cls(d=d, cutoff=cutoff, vacuum=True)

    @classmethod
    def from_pure(cls, coefficients, *, d, cutoff):
        # TODO: The result is not necessarily the pure state we might expect.
        # By applying a PNC operator to this or the true pure state, the result
        # is the same, but still.
        density_matrix = direct_sum(
            *(
                [np.outer(vector, vector) for vector in coefficients]
            )
        )

        return cls(density_matrix, d=d, cutoff=cutoff)

    def _apply(self, operator, modes):
        index = self._get_operator_index(modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        fock_operator = self._space.get_fock_operator(embedded_operator)

        self._density_matrix = (
            fock_operator @ self._density_matrix @ fock_operator.conjugate().transpose()
        )

    def _measure_particle_number(self):
        probabilities = self.fock_probabilities

        index = random.choices(range(len(self._space)), probabilities)[0]

        outcome_basis_vector = self._space[index]

        outcome = tuple(outcome_basis_vector)

        new_dm = np.zeros(shape=self._density_matrix.shape, dtype=complex)

        new_dm[index, index] = self._density_matrix[index, index]

        self._density_matrix = new_dm / probabilities[index]

        return outcome

    def _apply_creation_operator(self, modes):
        operator = self._space.get_creation_operator(modes)

        self._density_matrix = operator @ self._density_matrix @ operator.transpose()

        self.normalize()

    def _apply_annihilation_operator(self, modes):
        operator = self._space.get_annihilation_operator(modes)

        self._density_matrix = operator @ self._density_matrix @ operator.transpose()

        self.normalize()

    def __str__(self):
        ret = []

        for (index1, ket) in enumerate(self._space):
            for (index2, bra) in enumerate(self._space):
                matrix_element = self._density_matrix[index1, index2]
                if matrix_element == 0:
                    continue

                ret.append(
                    str(matrix_element)
                    + str(ket)
                    + bra.display_as_bra()
                )

        return " + ".join(ret)

    def __repr__(self):
        return str(self._density_matrix)

    @property
    def fock_probabilities(self):
        return np.diag(self._density_matrix).real

    @property
    def norm(self):
        return sum(self.fock_probabilities)

    def normalize(self):
        if np.isclose(self.norm, 0):
            raise RuntimeError("The norm of the state is 0.")

        self._density_matrix = self._density_matrix / self.norm
