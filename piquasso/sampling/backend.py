#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
from BoSS.BosonSamplingSimulator import BosonSamplingSimulator
from BoSS.simulation_strategies.GeneralizedCliffordsSimulationStrategy \
    import GeneralizedCliffordsSimulationStrategy

from piquasso.backend import Backend


class SamplingBackend(Backend):
    r"""A backend for fast boson sampling."""

    def phaseshift(self, operation):
        r"""Adds additional phase shifter to the effective interferometer.

        This can be interpreted as placing additional phase shifter (in the network)
        just before performing the sampling. This is realized by multiplying
        interferometer matrix with a matrix representing the phase shifter.

        The annihilation and creation operators on given mode are evolved in the
        following way:

        .. math::
            P(\phi) \hat{a}_k P(\phi)^\dagger = e^{i \phi} \hat{a}_k \\
            P(\phi) \hat{a}_k^\dagger P(\phi)^\dagger
                = e^{- i \phi} \hat{a}_k^\dagger

        Do note, that multiplication occurs only on one specified (as the argument)
        mode and the phase shifter acts as an identity on every other mode.
        """

        phi = operation.params[0]
        phase = np.exp(1j * phi)

        P = np.array([[phase]])

        self.state.multiple_interferometer_on_modes(P, operation.modes)

    def beamsplitter(self, operation):
        r"""Adds additional beam splitter to the effective interferometer.

        This can be interpreted as placing additional beam splitter (in the network)
        just before performing the sampling. This is realized by multiplying
        interferometer matrix with a matrix representing the phase shifter.

        The matrix representation of the beam splitters' operation on two modes is:

        .. math::
            B = \begin{bmatrix}
                t  & r^* \\
                -r & t
            \end{bmatrix},

        where
            :math:`t = \cos(\theta)` and
            :math:`r = e^{- i \phi} \sin(\theta)`.

        Do note, that multiplication occurs only on two specified (as the arguments)
        modes and the beam splitter acts as an identity on every other mode.
        """
        theta, phi = operation.params

        t = np.cos(theta)
        r = np.exp(-1j * phi) * np.sin(theta)

        B = np.array([
            [t, np.conj(r)],
            [-r, t]
        ])

        self.state.multiple_interferometer_on_modes(B, operation.modes)

    def interferometer(self, operation):
        J = operation.params[0]
        modes = operation.modes

        assert \
            J.shape == (len(modes), len(modes)), \
            "The number of qumodes should be equal to " \
            "the size of the interferometer matrix."

        self.state.multiple_interferometer_on_modes(J, modes)

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
