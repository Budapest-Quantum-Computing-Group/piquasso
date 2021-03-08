#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.core import _context
from piquasso.core.mixins import _PropertyMixin


class Operation(_PropertyMixin):
    def __init__(self, *params):
        """
        Args:
            *params: Variable length argument list.
        """
        self.params = params
        self.modes = None

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

        if self.modes:
            display_string += f" on modes {self.modes}"

        if self.params:
            display_string += f" with params {self.params}"

        return display_string


class ModelessOperation(Operation):
    def __init__(self, *params):
        super().__init__(*params)

        _context.current_program.operations.append(self)
