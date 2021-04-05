#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso.api.circuit import Circuit
from piquasso.api.result import Result


class GaussianCircuit(Circuit):

    def get_instruction_map(self):
        return {
            "Interferometer": self._passive_linear,
            "Beamsplitter": self._passive_linear,
            "Phaseshifter": self._passive_linear,
            "MachZehnder": self._passive_linear,
            "Fourier": self._passive_linear,
            "GaussianTransform": self._linear,
            "Squeezing": self._linear,
            "QuadraticPhase": self._linear,
            "Squeezing2": self._linear,
            "ControlledX": self._linear,
            "ControlledZ": self._linear,
            "Displacement": self._displacement,
            "PositionDisplacement": self._displacement,
            "MomentumDisplacement": self._displacement,
            "Graph": self._graph,
            "MeasureHomodyne": self._measure_homodyne,
            "MeasureHeterodyne": self._measure_dyne,
            "MeasureDyne": self._measure_dyne,
            "Vacuum": self._vacuum,
            "Mean": self._mean,
            "Covariance": self._covariance,
            "MeasureParticleNumber": self._measure_particle_number,
            "MeasureThreshold": self._measure_threshold,
        }

    def _passive_linear(self, instruction):
        self.state._apply_passive_linear(
            instruction._passive_representation,
            instruction.modes
        )

    def _linear(self, instruction):
        self.state._apply_linear(
            P=instruction._passive_representation,
            A=instruction._active_representation,
            modes=instruction.modes
        )

    def _displacement(self, instruction):
        self.state._apply_displacement(
            **instruction.params,
            modes=instruction.modes,
        )

    def _measure_homodyne(self, instruction):
        phi = instruction.params["phi"]
        modes = instruction.modes

        phaseshift = np.identity(len(modes)) * np.exp(- 1j * phi)

        self.state._apply_passive_linear(
            phaseshift,
            modes=modes,
        )

        samples = self.state._apply_generaldyne_measurement(
            detection_covariance=instruction.params["detection_covariance"],
            shots=instruction.params["shots"],
            modes=modes,
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _measure_dyne(self, instruction):
        samples = self.state._apply_generaldyne_measurement(
            detection_covariance=instruction.params["detection_covariance"],
            shots=instruction.params["shots"],
            modes=instruction.modes,
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _vacuum(self, instruction):
        self.state.reset()

    def _mean(self, instruction):
        self.state.mean = instruction.params["mean"]

    def _covariance(self, instruction):
        self.state.cov = instruction.params["cov"]

    def _measure_particle_number(self, instruction):
        samples = self.state._apply_particle_number_measurement(
            cutoff=instruction.params["cutoff"],
            shots=instruction.params["shots"],
            modes=instruction.modes,
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _measure_threshold(self, instruction):
        samples = self.state._apply_threshold_measurement(
            shots=instruction.params["shots"],
            modes=instruction.modes,
        )

        self.results.append(Result(instruction=instruction, samples=samples))

    def _graph(self, instruction):
        """
        TODO: Find a better solution for multiple operations.
        """
        instruction._squeezing.modes = instruction.modes
        instruction._interferometer.modes = instruction.modes

        self._linear(instruction._squeezing)
        self._passive_linear(instruction._interferometer)
