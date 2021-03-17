#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
from BoSS.BosonSamplingSimulator import BosonSamplingSimulator
from BoSS.simulation_strategies.GeneralizedCliffordsSimulationStrategy \
    import GeneralizedCliffordsSimulationStrategy

from piquasso.api.circuit import Circuit


class SamplingCircuit(Circuit):
    r"""A circuit for fast boson sampling."""

    def get_operation_map(self):
        return {
            "B": self._passive_linear,
            "R": self._passive_linear,
            "MZ": self._passive_linear,
            "F": self._passive_linear,
            "Sampling": self.sampling,
            "Interferometer": self._passive_linear,
        }

    def _passive_linear(self, operation):
        r"""Applies an interferometer to the circuit.

        This can be interpreted as placing another interferometer in the network, just
        before performing the sampling. This operation is realized by multiplying
        current effective interferometer matrix with new interferometer matrix.

        Do note, that new interferometer matrix works as interferometer matrix on
        qumodes (provided as the arguments) and as an identity on every other mode.
        """

        self.state._apply_passive_linear(
            operation._passive_representation,
            operation.modes,
        )

    def sampling(self, operation):
        params = operation.params

        simulation_strategy = \
            GeneralizedCliffordsSimulationStrategy(self.state.interferometer)
        sampling_simulator = BosonSamplingSimulator(simulation_strategy)

        initial_state = np.array(self.state.initial_state)
        shots = params[0]
        self.state.results = \
            sampling_simulator.get_classical_simulation_results(initial_state,
                                                                samples_number=shots)
