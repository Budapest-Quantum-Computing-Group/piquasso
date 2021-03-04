#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import random
import numpy as np

from piquasso.api.result import Result
from piquasso.api.circuit import Circuit


class PNCFockCircuit(Circuit):

    def get_operation_map(self):
        return {
            "PassiveTransform": self._apply,
            "B": self._apply,
            "R": self._apply,
            "MZ": self._apply,
            "F": self._apply,
            "MeasureParticleNumber": self._measure_particle_number,
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

        new_dm = np.zeros(shape=self.state._density_matrix.shape, dtype=complex)

        new_dm[index, index] = self.state._density_matrix[index, index]

        self.state._density_matrix = new_dm / probabilities[index]

        # TODO: Better way of providing results
        self.program.results.append(
            Result(measurement=operation, outcome=outcome)
        )
