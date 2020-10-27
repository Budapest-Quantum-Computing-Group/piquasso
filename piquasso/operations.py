#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Simple passive linear optical elements."""

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


class B(Operation):
    """Beamsplitter operation."""

    backends = {
        FockBackend: FockBackend.beamsplitter,
        GaussianBackend: GaussianBackend.beamsplitter,
        PassiveGaussianBackend: PassiveGaussianBackend.beamsplitter,
        SamplingBackend: SamplingBackend.beamsplitter
    }


class R(Operation):
    """Rotation or Phaseshifter operation."""

    backends = {
        FockBackend: FockBackend.phaseshift,
        GaussianBackend: GaussianBackend.phaseshift,
        PassiveGaussianBackend: PassiveGaussianBackend.phaseshift,
        SamplingBackend: SamplingBackend.phaseshift
    }


class D(Operation):
    """Displacement operation."""

    backends = {
        GaussianBackend: GaussianBackend.displacement,
    }


class Interferometer(Operation):
    """Interferometer"""

    backends = {
        SamplingBackend: SamplingBackend.interferometer
    }


class Sampling(Operation):
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
