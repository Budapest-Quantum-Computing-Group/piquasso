#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Implementation of the passive Gaussian-backends."""

import numpy as np

from ..backend import Backend


class PassiveGaussianBackend(Backend):

    def phaseshift(self, params, modes):
        r"""Performs a phase shifting on the quantum state.

        Evolves the annihilation and creation operators in the following way:

        .. math::
            P(\phi) \hat{a}_k P(\phi)^\dagger = e^{i \phi} \hat{a}_k \\
            P(\phi) \hat{a}_k^\dagger P(\phi)^\dagger
                = e^{- i \phi} \hat{a}_k^\dagger


        Args:
            params (tuple): An iterable with a single element corresponding
             to the angle of the phaseshifter.
            modes (tuple): An iterable with a single element corresponding
             to the mode of the phaseshifter.
        """

        phi = params[0]
        k = modes[0]

        phase_conj = np.conj(np.exp(1j * phi))

        self.state.C[k][:k] *= phase_conj
        self.state.C[k][k + 1:] *= phase_conj

        self.state.C[:, k] = np.conj(self.state.C[k])

    def beamsplitter(self, params, modes):
        r"""Applies a beamsplitter.

        Args:
            params (tuple): An iterable containing theta and phi, the angles
             of the beamsplitter.
            modes (tuple): An iterable containing the two distinct modes where
             the beamsplitter will operating on.
        """
        theta, phi = params
        i, j = modes

        phase = np.exp(1j * phi)
        sh = np.sin(theta)
        ch = np.cos(theta)
        sh2 = sh * sh
        ch2 = ch * ch
        shch = sh * ch

        Ci = np.copy(self.state.C[i])
        Cj = np.copy(self.state.C[j])

        self.state.C[i][i] = (
                ch2 * Ci[i]
                + phase * shch * Ci[j]
                + shch * np.conj(phase) * Cj[i] + sh2 * Cj[j]
        )
        self.state.C[i][j] = (
                -(shch * np.conj(phase) * Ci[i])
                + ch2 * Ci[j]
                - sh2 * np.conj(phase * phase) * Cj[i]
                + shch * np.conj(phase) * Cj[j]
        )
        self.state.C[j][i] = np.conj(self.state.C[i][j])
        self.state.C[j][j] = (
                sh2 * Ci[i]
                - phase * shch * Ci[j]
                - shch * np.conj(phase) * Cj[i] + ch2 * Cj[j]
        )

        idx = np.delete(np.arange(self.state.d), (i, j))
        self.state.C[i][idx] = ch * Ci[idx] + sh * np.conj(phase) * Cj[idx]
        self.state.C[j][idx] = -(phase * sh * Ci[idx]) + ch * Cj[idx]

        self.state.C[:, i] = np.conj(self.state.C[i])
        self.state.C[:, j] = np.conj(self.state.C[j])
