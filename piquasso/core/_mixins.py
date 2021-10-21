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
import copy
import typing
from typing import TypeVar, Generic

if typing.TYPE_CHECKING:
    from piquasso.api.program import Program
    from piquasso.api.mode import Q

_T = TypeVar("_T", float, complex)


class DictMixin(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dict_: dict) -> "DictMixin":
        """Creates an instance from a `dict` specified.

        Args:
            dict_ (dict): The desired instance in the format of a `dict`.
        """
        pass


class WeightMixin(abc.ABC, Generic[_T]):
    @property
    @abc.abstractmethod
    def params(self) -> dict:
        pass

    def __mul__(self, coefficient: _T) -> "WeightMixin":
        self.params["coefficient"] *= coefficient
        return self

    __rmul__ = __mul__

    def __truediv__(self, coefficient: _T) -> "WeightMixin":
        return self.__mul__(1 / coefficient)


class RegisterMixin(abc.ABC):
    @abc.abstractmethod
    def _apply_to_program_on_register(self, program: "Program", register: "Q") -> None:
        """Applies the current object to the specifed program on its specified register.

        Args:
            program (Program): [description]
            register (Q): [description]
        """
        pass

    def copy(self) -> "RegisterMixin":
        """Copies the current object with :func:`copy.deepcopy`.

        Returns:
            A deepcopy of the current object.
        """
        return copy.deepcopy(self)


class CodeMixin(abc.ABC):
    @abc.abstractmethod
    def _as_code(self) -> str:
        pass


class ScalingMixin(abc.ABC):
    @abc.abstractmethod
    def _autoscale(self) -> None:
        pass
