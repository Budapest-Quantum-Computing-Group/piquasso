#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.api.state import State

from piquasso._math import fock, direct_sum

from .circuit import PNCFockCircuit


class PNCFockState(State):
    _circuit_class = PNCFockCircuit

    def __init__(self, density_matrix, *, d, cutoff):
        self._density_matrix = density_matrix
        self._space = fock.FockSpace(
            d=d,
            cutoff=cutoff,
        )

    @classmethod
    def create_vacuum(cls, *, d, cutoff):
        array = np.zeros((fock.FockSpace(d=d, cutoff=cutoff).cardinality, ) * 2)

        return cls(
            array,
            d=d, cutoff=cutoff
        )

    @classmethod
    def from_pure(cls, coefficients, *, d, cutoff):
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

    def __str__(self):
        basis_vectors = self._space.basis_vectors

        ret = []

        for (index1, ket) in enumerate(basis_vectors):
            for (index2, bra) in enumerate(basis_vectors):
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
