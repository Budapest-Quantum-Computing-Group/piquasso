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
        r"""
        Multiplies the interferometer with the matrix representation of the phaseshifter

        Args:
            params (tuple): An iterable with a single element, which corresponds to the
                angle of the phaseshifter.
            modes (tuple): An iterable with a single element, which corresponds to the
                mode of the phaseshifter.
        """
        phi = params[0]
        phase = np.exp(1j * phi)

        P = np.array([[phase]])

        self.state.multiple_interferometer_on_modes(P, modes)

    def beamsplitter(self, params, modes):
        r"""
        Multiplies the interferometer with the matrix representation of the beamsplitter

        The matrix representation of the beamsplitter operation is:

        .. math::
            B = \begin{bmatrix}
                t  & r^* \\
                -r & t
            \end{bmatrix},

        where
            :math:`t = \cos(\theta)` and
            :math:`r = e^{- i \phi} \sin(\theta)`.

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
        r"""
        Multiplies the interferometer of the state with the interferometer given in
        `params` in the qumodes specified in `modes`.

        Args:
            params (tuple): A tuple containing a single square matrix that represents
                the interferometer.
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
        r"""Simulates a boson sampling using generalized Clifford&Clifford algorithm
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
