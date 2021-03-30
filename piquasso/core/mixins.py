#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc
import copy


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


class _RegisterMixin(abc.ABC):
    @abc.abstractmethod
    def apply_to_program_on_register(self, *, program, register):
        """Applies the current object to the specifed program on its specified register.

        Args:
            program (Program): [description]
            register (Q): [description]
        """
        pass

    def copy(self):
        """Copies the current object with :func:`copy.deepcopy`.

        Returns:
            A deepcopy of the current object.
        """
        return copy.deepcopy(self)
