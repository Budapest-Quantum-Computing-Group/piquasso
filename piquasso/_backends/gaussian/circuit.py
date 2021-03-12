#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.api.circuit import Circuit
from piquasso.api.result import Result


class GaussianCircuit(Circuit):

    def get_operation_map(self):
        return {
            "PassiveTransform": self._passive_linear,
            "B": self._passive_linear,
            "R": self._passive_linear,
            "MZ": self._passive_linear,
            "F": self._passive_linear,
            "GaussianTransform": self._linear,
            "S": self._linear,
            "P": self._linear,
            "S2": self._linear,
            "CX": self._linear,
            "CZ": self._linear,
            "D": self._displacement,
            "X": self._displacement,
            "Z": self._displacement,
        }

    def _passive_linear(self, operation):
        self.state._apply_passive_linear(
            operation._passive_representation,
            operation.modes
        )

    def _linear(self, operation):
        self.state._apply_linear(
            P=operation._passive_representation,
            A=operation._active_representation,
            modes=operation.modes
        )

    def _displacement(self, operation):
        params = operation.params
        modes = operation.modes

        if len(params) == 1:
            alpha = params[0]
        else:
            r, phi = params
            alpha = r * np.exp(1j * phi)

        mode = modes[0]

        m_copy = np.copy(self.state.m)

        self.state.m[mode] += alpha

        self.state.C[mode, mode] += (
            2 * (np.conj(alpha) * m_copy[mode]).real
            + np.conj(alpha) * alpha
        )

        self.state.G[mode, mode] += 2 * alpha * m_copy[mode] + alpha * alpha

        other_modes = np.delete(np.arange(self.state.d), modes)

        self.state.C[other_modes, mode] += alpha * np.conj(m_copy[other_modes])
        self.state.G[other_modes, mode] += alpha * m_copy[other_modes]

        self.state.C[mode, :] = self.state.C[:, mode].conjugate().transpose()
        self.state.G[mode, :] = self.state.G[:, mode].transpose()

    def _measure_dyne(self, operation):
        outcome = self.state._apply_generaldyne_measurement(
            detection_covariance=operation.params[0],
            modes=operation.modes,
        )

        # TODO: Better way of providing results
        self.program.results.append(
            Result(measurement=operation, outcome=outcome)
        )