#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import numpy as np

from piquasso.core import _mixins


class State(_mixins.PropertyMixin, _mixins.RegisterMixin, _mixins.CodeMixin, abc.ABC):
    """The base class from which all `*State` classes are derived.

    Attributes:
        circuit_class (~piquasso.api.circuit.Circuit):
            Class attribute for specifying corresponding circuit. The circuit is
            responsible to execute the specified instructions on the :class:`State`
            instance.
        d (int): Instance attribute specifying the number of modes.
    """

    circuit_class = None
    d: int = None

    def _apply_to_program_on_register(self, program, register):
        program.state = self.copy()

    @classmethod
    def from_properties(cls, properties: dict):
        return cls(**properties)

    def _as_code(self):
        return f"pq.Q() | pq.{self.__class__.__name__}(d={self.d})"

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
