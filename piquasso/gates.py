#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Simple passive linear optical elements."""

from piquasso.backend import FockBackend
from piquasso.context import Context


class Gate:
    backends = {}

    def __init__(self, *params):
        self.params = params

    def resolve_method_for_backend(self):
        method = self.backends.get(Context.current_program.backend.__class__)

        if not method:
            raise NotImplementedError("No such operation implemented on this backend.")

        return method


class B(Gate):
    """Beamsplitter gate."""

    backends = {
        FockBackend: FockBackend.beamsplitter
    }
