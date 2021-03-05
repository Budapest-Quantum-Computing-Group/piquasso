#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import random
import numpy as np

from piquasso.api.result import Result
from piquasso.api.circuit import Circuit


class PureFockCircuit(Circuit):

    def get_operation_map(self):
        return {
            "PassiveTransform": self._apply,
            "B": self._apply,
            "R": self._apply,
            "MZ": self._apply,
            "F": self._apply,
            "MeasureParticleNumber": self._measure_particle_number,
            "Number": self._number,
        }

    def _apply(self, operation):
        self.state._apply(
            operator=operation._passive_representation,
            modes=operation.modes
        )

    def _measure_particle_number(self, operation):
        basis_vectors = self.state._space.basis_vectors

        probabilities = self.state.fock_probabilities

        index = random.choices(range(len(basis_vectors)), probabilities)[0]

        outcome_basis_vector = basis_vectors[index]

        outcome = tuple(outcome_basis_vector)

        new_state_vector = np.zeros(
            shape=self.state._state_vector.shape,
            dtype=complex,
        )

        new_state_vector[index] = self.state._state_vector[index]

        self.state._density_matrix = new_state_vector / probabilities[index]

        # TODO: Better way of providing results
        self.program.results.append(
            Result(measurement=operation, outcome=outcome)
        )

    def _number(self, operation):
        occupation_numbers = operation.params[0]
        coefficient = operation.params[1]

        index = self.state._space.get_index_by_occupation_basis(occupation_numbers)

        self.state._state_vector[index] = coefficient
