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
from typing import Tuple

import numpy as np

from .mode import Q
from piquasso.core import _mixins, _registry


class Instruction(_mixins.DictMixin, _mixins.RegisterMixin, _mixins.CodeMixin):
    """
    Args:
        *params: Variable length argument list.
    """

    def __init__(self, *, params: dict = None, extra_params: dict = None) -> None:
        self._params: dict = params or dict()

        self._extra_params: dict = extra_params or dict()

    @property
    def params(self) -> dict:
        return self._params

    @property
    def _all_params(self) -> dict:
        return {**self._params, **self._extra_params}

    def _as_code(self) -> str:
        if hasattr(self, "modes"):
            mode_string = ", ".join([str(mode) for mode in self.modes])
        else:
            mode_string = ""

        if hasattr(self, "params"):
            params_string = "{}".format(
                ", ".join(
                    [
                        f"{key}={self._param_repr(value)}"
                        for key, value
                        in self.params.items()
                    ]
                )
            )
        else:
            params_string = ""

        return f"pq.Q({mode_string}) | pq.{self.__class__.__name__}({params_string})"

    @staticmethod
    def _param_repr(value) -> str:
        if isinstance(value, np.ndarray):
            return "np." + repr(value)

        return value

    def on_modes(self, *modes: int) -> "Instruction":
        self.modes: Tuple[int, ...] = modes
        return self

    def _apply_to_program_on_register(self, program, register: Q) -> None:
        program.instructions.append(self.on_modes(*register.modes))

    @classmethod
    def from_dict(cls, dict_: dict) -> "Instruction":
        """Creates an :class:`Instruction` instance from a dict specified.

        Args:
            dict_ (dict):
                The desired :class:`Instruction` instance in the format of a dict.

        Returns:
            Instruction:
                An :class:`Instruction` initialized using the specified dict.
        """

        class_ = _registry.get_class(dict_["type"])

        instruction = class_(**dict_["attributes"]["constructor_kwargs"])

        instruction.modes = dict_["attributes"]["modes"]

        return instruction

    def __repr__(self) -> str:
        if hasattr(self, "modes"):
            modes = "modes={}".format(self.modes)
        else:
            modes = ""

        if getattr(self, "params") != {}:
            params = "{}, ".format(
                ", ".join(
                    [f"{key}={value}" for key, value in self.params.items()]
                )
            )
        else:
            params = ""

        classname = self.__class__.__name__

        return f"<pq.{classname}({params}{modes})>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instruction):
            return False
        return (
            self.modes == other.modes
            and
            self.params == other.params
        )
