#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.api.circuit import Circuit


class GaussianCircuit(Circuit):

    def get_operation_map(self):
        return {
            "PassiveTransform": self._apply_passive,
            "B": self._apply_passive,
            "R": self._apply_passive,
            "MZ": self._apply_passive,
            "F": self._apply_passive,
            "GaussianTransform": self._apply,
            "S": self._apply,
            "P": self._apply,
            "S2": self._apply,
            "D": self.displacement
        }

    def _apply_passive(self, operation):
        self.state.apply_passive(
            operation._passive_representation,
            operation.modes
        )

    def _apply(self, operation):
        self.state.apply_active(
            P=operation._passive_representation,
            A=operation._active_representation,
            modes=operation.modes
        )

    def displacement(self, operation):
        params = operation.params
        modes = operation.modes

        if len(params) == 1:
            alpha = params[0]
        else:
            r, phi = params
            alpha = r * np.exp(1j * phi)

        mode = modes[0]

        mean_copy = np.copy(self.state.m)

        self.state.m[mode] += alpha

        self.state.C[mode, mode] += (
            alpha * np.conj(mean_copy[mode])
            + np.conj(alpha) * mean_copy[mode]
            + np.conj(alpha) * alpha
        )

        self.state.G[mode, mode] += 2 * alpha * mean_copy[mode] + alpha * alpha

        other_modes = np.delete(np.arange(self.state.d), modes)

        self.state.C[other_modes, mode] += alpha * mean_copy[other_modes]
        self.state.C[mode, other_modes] += np.conj(alpha) * mean_copy[other_modes]

        self.state.G[other_modes, mode] += alpha * mean_copy[other_modes]
        self.state.G[mode, other_modes] += alpha * mean_copy[other_modes]
