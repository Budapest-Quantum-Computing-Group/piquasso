#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Simple passive linear optical elements."""

from piquasso.context import Context

from piquasso.fock.backend import FockBackend
from piquasso.gaussian.backend import GaussianBackend
from piquasso.passivegaussian.backend import PassiveGaussianBackend


class Gate:
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
        """Maps the name of a BlackBird gate into the represented class

        Args:
            op (string): the representation of a gate in BlackBird

        Returns:
            Gate: subclass of :class:`Gate` that the argument represents
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
            }[op]


class B(Gate):
    """Beamsplitter gate."""

    backends = {
        FockBackend: FockBackend.beamsplitter,
        GaussianBackend: GaussianBackend.beamsplitter,
        PassiveGaussianBackend: PassiveGaussianBackend.beamsplitter
    }


class R(Gate):
    """Rotation or Phaseshifter gate."""

    backends = {
        FockBackend: FockBackend.phaseshift,
        GaussianBackend: GaussianBackend.phaseshift,
        PassiveGaussianBackend: PassiveGaussianBackend.phaseshift
    }


class D(Gate):
    """Displacement gate."""

    backends = {
        GaussianBackend: GaussianBackend.displacement,
    }
