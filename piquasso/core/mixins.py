#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc


class _PropertyMixin(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def from_properties(cls, properties):
        """Creates an instance from a mapping specified.

        Args:
            properties (collections.Mapping):
                The desired instance in the format of a mapping.
        """
        pass


class _WeightMixin:
    def __mul__(self, coefficient):
        self.params["coefficient"] *= coefficient
        return self

    __rmul__ = __mul__

    def __truediv__(self, coefficient):
        return self.__mul__(1 / coefficient)
