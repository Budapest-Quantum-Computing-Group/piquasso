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

    def phaseshift(self, params, modes):
        """Adds additional phase shifter to the effective interferometer.

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

        Args:
            params (tuple): An iterable with a single element, which corresponds to the
                angle of the phase shifter.
            modes (tuple): An iterable with a single element, which corresponds to the
                mode of the phase shifter.
        """

        phi = params[0]
        phase = np.exp(1j * phi)

        P = np.array([[phase]])

        self.state.multiple_interferometer_on_modes(P, modes)

    def beamsplitter(self, params, modes):
        """Adds additional beam splitter to the effective interferometer.

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

        Args:
            params (tuple): Angle parameters :math:`\phi` and :math:`\theta` for the
                beamsplitter operation.
            modes (tuple): Distinct positive integer values which are used to represent
                qumodes.
        """
        phi, theta = params

        t = np.cos(theta)
        r = np.exp(-1j * phi) * np.sin(theta)

        B = np.array([
            [t, np.conj(r)],
            [-r, t]
        ])

        self.state.multiple_interferometer_on_modes(B, modes)

    def interferometer(self, params, modes):
        """Adds additional interferometer to the effective interferometer.

        This can be interpreted as placing another interferometer in the network, just
        before performing the sampling. This operation is realized by multiplying
        current effective interferometer matrix with new interferometer matrix.

        Do note, that new interferometer matrix works as interferometer matrix on
        qumodes (provided as the arguments) and as an identity on every other mode.

        Args:
            params (tuple): A tuple containing a single square matrix that represents
                the additional interferometer.
            modes (tuple): Distinct positive integer values which are used to represent
                qumodes.
        """
        J = params[0]

        assert \
            J.shape == (len(modes), len(modes)), \
            "The number of qumodes should be equal to " \
            "the size of the interferometer matrix."

        self.state.multiple_interferometer_on_modes(J, modes)

    def sampling(self, params):
        """Simulates a boson sampling using generalized Clifford&Clifford algorithm
        from [Brod, Oszmaniec 2020].

        This method assumes that initial_state is given in the second quantization
        description (mode occupation). BoSS requires input states as numpy arrays,
        therefore the state is prepared as such structure.

        Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
        as it allows effective simulation of broader range of input states than original
        algorithm.

        Args:
            params (tuple): A tuple with a single element corresponding to number of
                samples for the experiment.
        """
        simulation_strategy = \
            GeneralizedCliffordsSimulationStrategy(self.state.interferometer)
        sampling_simulator = BosonSamplingSimulator(simulation_strategy)

        initial_state = np.array(self.state.initial_state)
        shots = params[0]
        self.state.results = \
            [sampling_simulator.get_classical_simulation_results(initial_state)
             for _ in range(shots)]
