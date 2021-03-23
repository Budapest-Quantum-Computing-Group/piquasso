#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.api.circuit import Circuit
from piquasso.api.result import Result


class GaussianCircuit(Circuit):

    def get_operation_map(self):
        return {
            "Interferometer": self._passive_linear,
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
            "MeasureHomodyne": self._measure_dyne,
            "MeasureHeterodyne": self._measure_dyne,
            "MeasureDyne": self._measure_dyne,
            "Vacuum": self._vacuum,
            "Mean": self._mean,
            "Covariance": self._covariance,
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
        self.state._apply_displacement(
            alpha=operation.params[0],
            mode=operation.modes[0],
        )

    def _measure_dyne(self, operation):
        outcomes = self.state._apply_generaldyne_measurement(
            detection_covariance=operation.params[0],
            modes=operation.modes,
            shots=operation.params[1],
        )

        # TODO: Better way of providing results
        for outcome in outcomes:
            self.program.results.append(
                Result(measurement=operation, outcome=outcome)
            )

    def _vacuum(self, operation):
        self.state.reset()

    def _mean(self, operation):
        self.state.mean = operation.params[0]

    def _covariance(self, operation):
        self.state.cov = operation.params[0]
