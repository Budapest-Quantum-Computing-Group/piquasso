#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc
import numpy as np

from piquasso.core.mixins import _PropertyMixin


class State(_PropertyMixin, abc.ABC):
    circuit_class = None
    d: int = None

    @classmethod
    def from_properties(cls, properties):
        return cls(**properties)

    @staticmethod
    def _get_operator_index(modes):
        """
        Note:
            For indexing of numpy arrays, see
            https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
        """

        transformed_columns = np.array([modes] * len(modes))
        transformed_rows = transformed_columns.transpose()

        return transformed_rows, transformed_columns

    def _get_auxiliary_modes(self, modes):
        return np.delete(np.arange(self.d), modes)

    @staticmethod
    def _get_auxiliary_operator_index(modes, auxiliary_modes):
        auxiliary_rows = np.array([modes] * len(auxiliary_modes)).transpose()

        return auxiliary_rows, auxiliary_modes
