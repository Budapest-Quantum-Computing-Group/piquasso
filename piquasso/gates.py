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


class B(Gate):
    """Beamsplitter gate."""

    backends = {
        FockBackend: FockBackend.beamsplitter,
        GaussianBackend: GaussianBackend.beamsplitter,
        PassiveGaussianBackend: PassiveGaussianBackend.beamsplitter
    }
