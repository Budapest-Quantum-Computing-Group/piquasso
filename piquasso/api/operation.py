#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.core.mixins import _PropertyMixin


class Operation(_PropertyMixin):
    """
    Args:
        *params: Variable length argument list.
    """

    def __init__(self, *params):
        self._set_params(*params)

    def _set_params(self, *params):
        self.params = params

    @classmethod
    def from_properties(cls, properties):
        """Creates an `Operation` instance from a mapping specified.

        Args:
            properties (collections.Mapping):
                The desired `Operator` instance in the format of a mapping.

        Returns:
            Operator: An `Operator` initialized using the specified mapping.
        """

        operation = cls(**properties["params"])

        operation.modes = properties["modes"]

        return operation

    def __repr__(self):
        display_string = self.__class__.__name__

        if getattr(self, "modes", None):
            display_string += f" on modes {self.modes}"

        if getattr(self, "params", None):
            display_string += f" with params {self.params}"

        return display_string
