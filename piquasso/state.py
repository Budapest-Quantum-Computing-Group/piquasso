#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from ._mixins import PropertyMixin


class State(PropertyMixin):
    _circuit_class = None

    @classmethod
    def from_properties(cls, properties):
        return cls(**properties)

    @staticmethod
    def _get_operator_index(modes):
        transformed_columns = np.array([modes] * len(modes))
        transformed_rows = transformed_columns.transpose()

        return transformed_rows, transformed_columns

    def _get_auxiliary_modes(self, modes):
        return np.delete(np.arange(self.d), modes)

    @staticmethod
    def _get_auxiliary_operator_index(modes, auxiliary_modes):
        auxiliary_rows = np.array([modes] * len(auxiliary_modes)).transpose()

        return auxiliary_rows, auxiliary_modes
