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

import typing
from typing import Tuple, Any, Type, Dict

import numpy as np

from .mode import Q
from piquasso.core import _mixins
from piquasso.api.errors import PiquassoException

if typing.TYPE_CHECKING:
    from piquasso.api.program import Program


class Instruction(_mixins.DictMixin, _mixins.RegisterMixin, _mixins.CodeMixin):
    """
    Base class for all instructions.

    Args:
        params: Mapping of parameters specified by the users.
        extra_params: Mapping of extra parameters, typically calculated ones.
    """

    _subclasses: Dict[str, Type["Instruction"]] = {}

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
                        for key, value in self.params.items()
                    ]
                )
            )
        else:
            params_string = ""

        return f"pq.Q({mode_string}) | pq.{self.__class__.__name__}({params_string})"

    @staticmethod
    def _param_repr(value: Any) -> str:
        if isinstance(value, np.ndarray):
            return "np." + repr(value)

        return value

    def on_modes(self, *modes: int) -> "Instruction":
        self.modes: Tuple[int, ...] = modes
        return self

    def _apply_to_program_on_register(self, program: "Program", register: Q) -> None:
        program.instructions.append(self.on_modes(*register.modes))

    @classmethod
    def from_dict(cls, dict_: dict) -> "Instruction":
        """Creates an :class:`Instruction` instance from a dict specified.

        Args:
            dict_ (dict):
                The desired :class:`Instruction` instance in the format of a `dict`.

        Returns:
            Instruction:
                An :class:`Instruction` initialized using the specified `dict`.
        """

        class_ = cls.get_subclass(dict_["type"])

        instruction = class_(**dict_["attributes"]["constructor_kwargs"])

        instruction.modes = dict_["attributes"]["modes"]

        return instruction

    @classmethod
    def set_subclass(cls, instruction: Type["Instruction"]) -> None:
        """Registers a class in the instruction subclass map.

        This is meaningful in contexts when one has multiple instructions with the same
        name.

        Example:
            When one creates a custom beamsplitter with name `Beamsplitter` and
            subclasses :class:`~piquasso.instructions.gates.Beamsplitter`, then for e.g.
            executing a Blackbird code will be performed with the custom one, not the
            original one. When one wants to use the original one in this case, one can
            reset it with this method.

        Args:
            instruction (Type[Instruction]): The instruction class to be registered.

        Raises:
            PiquassoException:
                When the class is not actually an instance of :class:`Insruction`.
        """

        if not issubclass(instruction, Instruction):
            raise PiquassoException(
                f"The instruction '{instruction}' needs to be a subclass of "
                "'pq.Instruction'."
            )

        cls._subclasses[instruction.__name__] = instruction

    @classmethod
    def get_subclass(cls, name: str) -> Type["Instruction"]:
        """Returns the instruction subclass specified by its name.

        Returns:
            Type[Instruction]: The instruction class.
        """

        return cls._subclasses[name]

    def __repr__(self) -> str:
        if hasattr(self, "modes"):
            modes = "modes={}".format(self.modes)
        else:
            modes = ""

        if getattr(self, "params") != {}:
            params = "{}, ".format(
                ", ".join([f"{key}={value}" for key, value in self.params.items()])
            )
        else:
            params = ""

        classname = self.__class__.__name__

        return f"<pq.{classname}({params}{modes})>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instruction):
            return False
        return self.modes == other.modes and self.params == other.params

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        cls.set_subclass(cls)


class Preparation(Instruction):
    """Base class for preparations."""


class Gate(Instruction):
    """Base class for gates."""


class Measurement(Instruction):
    r"""Base class for measurements."""
