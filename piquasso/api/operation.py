#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.core.mixins import _PropertyMixin


class Operation(_PropertyMixin):
    """
    Args:
        *params: Variable length argument list.
    """

    def __init__(self, **params):
        self._set_params(**params)

    def _set_params(self, **params):
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
        if hasattr(self, "modes"):
            modes = "modes={}".format(self.modes)
        else:
            modes = ""

        if hasattr(self, "params"):
            params = "{}".format(", ".join([str(param) for param in self.params]))
        else:
            params = ""

        classname = self.__class__.__name__

        return f"<pq.{classname}({params}, {modes})>"
