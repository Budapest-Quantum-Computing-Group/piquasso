#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Simple passive linear optical elements."""

import numpy as np

from piquasso.context import Context

from piquasso.fock.backend import FockBackend
from piquasso.gaussian.backend import GaussianBackend
from piquasso.passivegaussian.backend import PassiveGaussianBackend
from piquasso.sampling.backend import SamplingBackend


class Operation:
    backends = {}

    def __init__(self, *params):
        self.params = params

    def resolve_method_for_backend(self):
        method = self.backends.get(Context.current_program.backend.__class__)

        if not method:
            raise NotImplementedError(
                "No such operation implemented on this backend."
            )

        return method

    @staticmethod
    def blackbird_op_to_gate(op):
        """Maps the name of a BlackBird operation into the represented class

        Args:
            op (string): the representation of a operation in BlackBird

        Returns:
            Operation: subclass of :class:`Operation` that the argument represents
        """
        return \
            {
                "Dgate": D,
                "Xgate": None,
                "Zgate": None,
                "Sgate": None,
                "Pgate": None,
                "Vgate": None,
                "Kgate": None,
                "Rgate": R,
                "BSgate": B,
                "MZgate": None,
                "S2gate": None,
                "CXgate": None,
                "CZgate": None,
                "CKgate": None,
                "Fouriergate": None
            }.get(op)


class ModelessOperation(Operation):

    def __init__(self, *params):
        super().__init__(*params)

        Context.current_program.instructions.append(
            {
                'params': self.params,
                'op': self.resolve_method_for_backend()
            }
        )


class B(Operation):
    """Beamsplitter operation."""

    backends = {
        FockBackend: FockBackend.beamsplitter,
        GaussianBackend: GaussianBackend.beamsplitter,
        PassiveGaussianBackend: PassiveGaussianBackend.beamsplitter,
        SamplingBackend: SamplingBackend.beamsplitter
    }

    def __init__(self, theta=0., phi=np.pi / 4):
        r"""Beamsplitter operation

        Args:
            phi (float): Phase angle of the beamsplitter.
                (defaults to :math:`\phi = \pi/2` that gives a symmetric beamsplitter)
            theta (float): The transmittivity angle of the beamsplitter.
                (defaults to :math:`\theta=\pi/4` that gives a 50-50 beamsplitter)
        """
        super().__init__(phi, theta)


class R(Operation):
    """Rotation or Phaseshifter operation."""

    backends = {
        FockBackend: FockBackend.phaseshift,
        GaussianBackend: GaussianBackend.phaseshift,
        PassiveGaussianBackend: PassiveGaussianBackend.phaseshift,
        SamplingBackend: SamplingBackend.phaseshift
    }

    def __init__(self, phi):
        r"""Rotation or Phaseshifter operation.

        Args:
            phi (float): The angle of the rotation.
        """
        super().__init__(phi)


class D(Operation):
    """Displacement operation."""

    backends = {
        GaussianBackend: GaussianBackend.displacement,
    }

    def __init__(self, *, alpha=None, r=None, phi=None):
        r"""Phase space displacement operation.

        One must either specify :param:`alpha` only, or the combination of :param:`r`
        and :param:`phi`.

        When :param:`r` and :param:`phi` are the given parameters, `alpha` is calculated
        via:

        .. math:
            \alpha = r \exp(i \phi)

        Args:
            alpha (complex): The displacement.
            r (float): The displacement magnitude.
            phi (float): The displacement angle.
        """
        assert \
            alpha is not None and r is None and phi is None \
            or \
            alpha is None and r is not None and phi is not None, \
            "Either specify 'alpha' only, or the combination of 'r' and 'phi'."

        if alpha is None:
            alpha = r * np.exp(1j * phi)

        super().__init__(alpha)


class Interferometer(Operation):
    """Interferometer"""

    backends = {
        SamplingBackend: SamplingBackend.interferometer
    }

    @staticmethod
    def _is_square(matrix):
        shape = matrix.shape
        return len(shape) == 2 and shape[0] == shape[1]

    def __init__(self, interferometer_matrix):
        r"""Interferometer

        Args:
            interferometer_matrix (np.array): A square matrix representing the
                interferometer
        """
        assert \
            self._is_square(interferometer_matrix), \
            "The interferometer matrix should be a square matrix."
        super().__init__(interferometer_matrix)


class Sampling(ModelessOperation):
    r"""Boson Sampling"""

    backends = {
        SamplingBackend: SamplingBackend.sampling
    }

    def __init__(self, shots=1):
        r"""Boson Sampling

        Args:
            shots (int): A positive integer value representing number of samples for the
                experiment
        """
        assert \
            shots > 0 and isinstance(shots, int),\
            "The number of shots should be a positive integer."
        super().__init__(shots)
