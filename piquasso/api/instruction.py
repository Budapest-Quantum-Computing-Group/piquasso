#
# Copyright 2021-2025 Budapest Quantum Computing Group
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
from typing import Optional, Tuple, Any, Type, Dict, Callable, Union

import numpy as np

from .mode import Q
from piquasso.core import _mixins
from piquasso.api.exceptions import PiquassoException, InvalidProgram, InvalidParameter

if typing.TYPE_CHECKING:
    from piquasso.api.program import Program
    from piquasso.api.connector import BaseConnector


class Instruction(_mixins.DictMixin, _mixins.RegisterMixin, _mixins.CodeMixin):
    """
    Base class for all instructions.

    Args:
        params: Mapping of parameters specified by the users.
    """

    NUMBER_OF_MODES: Optional[int] = None

    _subclasses: Dict[str, Type["Instruction"]] = {}

    def __init__(self, *, params: Optional[dict] = None) -> None:
        self._params: dict = params or dict()

        self._condition: Optional[Callable] = None

        is_unresolved = callable
        self._unresolved_params = {
            name: param for name, param in self._params.items() if is_unresolved(param)
        }

    def _validate(self, connector: "BaseConnector") -> None:
        pass

    @property
    def params(self) -> dict:
        """The parameters of the instruction as a `dict`."""
        return self._params

    def _get_computed_params(self, connector: "BaseConnector") -> dict:
        return dict()

    def _get_all_params(self, connector: "BaseConnector") -> dict:
        return {**self._params, **self._get_computed_params(connector)}

    def _is_resolved(self):
        return self._unresolved_params == dict()

    def _resolve_params(self, outcomes):
        _resolved_params = {}

        for name, unresolved_param in self._unresolved_params.items():
            try:
                resolved_param = unresolved_param(outcomes)
            except Exception as e:
                raise InvalidParameter(
                    f"An error occurred when resolving the parameter "
                    f"'{name}' for the instruction '{self}': {e}\n"
                    f"Make sure that the parameter is a callable that takes the "
                    f"tuple of measurement outcomes and returns a float."
                ) from e

            _resolved_params[name] = resolved_param

        self._params.update(_resolved_params)

    def _unresolve_params(self):
        self._params.update(self._unresolved_params)

    @property
    def modes(self) -> Tuple[int, ...]:
        """The qumodes on which the instruction is applied."""
        return getattr(self, "_modes", tuple())

    @modes.setter
    def modes(self, value: Tuple[int, ...]) -> None:
        self._validate_modes(value)
        self._modes = value

    @property
    def condition(self) -> Optional[Callable]:
        """The condition associated with the instruction."""
        return self._condition

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

        if self._condition is not None:
            raise PiquassoException(
                f"Cannot convert a conditioned instruction to code. "
                f"instruction: {self}"
            )

        return f"pq.Q({mode_string}) | pq.{self.__class__.__name__}({params_string})"

    @staticmethod
    def _param_repr(value: Any) -> str:
        if isinstance(value, np.ndarray):
            return "np." + repr(value)

        return value

    def on_modes(self, *modes: int) -> "Instruction":
        """Specifies the modes on which the state should be executed.

        Returns:
            Instruction: The same instruction with the modes specified.
        """
        if modes is not tuple():
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

        if getattr(self, "params", {}) != {}:
            params = "{}, ".format(
                ", ".join([f"{key}={value}" for key, value in self.params.items()])
            )
        else:
            params = ""

        if self._condition:
            condition = f", condition={self._condition}"
        else:
            condition = ""

        classname = self.__class__.__name__

        return f"{classname}({params}{modes}{condition})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instruction):
            return False
        return self.modes == other.modes and self.params == other.params

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        cls.set_subclass(cls)

    def _validate_modes(self, modes):
        if self.NUMBER_OF_MODES is not None and len(modes) != self.NUMBER_OF_MODES:
            raise InvalidProgram(
                f"The modes '{modes}' got specifed for the instruction '{self}', but "
                f"exactly '{self.NUMBER_OF_MODES}' mode needs to be specified. "
                f"Concretely, the total number of modes specified for this instruction "
                f"is 'len(modes) == len({modes}) == {len(modes)} != "
                f"{self.NUMBER_OF_MODES}'."
            )

    def when(self, condition: Callable) -> "Instruction":
        """Makes the execution of the instruction conditional on a callable object.

        The callable object takes a single argument - the tuple of measurement outcomes
        - and returns a boolean indicating whether the condition evaluates to ``True``.
        The condition must be met for the instruction to be executed.

        The measurement outcomes are provided as a tuple, in the order which the
        measurements are defined. For example, `x[0]` refers to the first outcome, and
        `x[-1]` to the last outcome.

        Example usage::
            with pq.Program() as program:
                pq.Q(0, 1) | pq.StateVector([0, 2]) * np.sqrt(1 / 2)
                pq.Q(0, 1) | pq.StateVector([2, 0]) * np.sqrt(1 / 2)

                pq.Q(1)    | pq.ParticleNumberMeasurement().on_modes(1)
                pq.Q(0)    | pq.Squeezing(r=r).on_modes(0).when(lambda x: x[-1] == 2)

                pq.Q(0, 1) | pq.Beamsplitter5050().when(lambda x: x[0] > 0)

        Raises:
            PiquassoException:
                If the instruction is already conditioned on another callable, or if the
                provided condition is not callable.

        Args:
            condition (Callable):
                A callable that takes the measurement outcomes and returns a boolean.
        """
        if self._condition is not None:
            raise PiquassoException(
                f"The instruction '{self}' is already conditioned on "
                f"'{self._condition}'."
            )

        if not callable(condition):
            raise PiquassoException(
                f"The condition '{condition}' needs to be a callable that takes the "
                f"measurement outcomes and returns a boolean."
            )

        self._condition = condition

        return self

    def _is_condition_met(self, outcomes: Tuple[Union[int, float], ...]) -> bool:
        """Checks if the condition associated with the instruction is met.

        Raises:
            PiquassoException:
                If an error occurs when evaluating the condition.

        Returns:
            bool: True if the condition is met, False otherwise.
        """
        if not self._condition:
            return True

        try:
            return self._condition(outcomes)
        except Exception as e:
            raise PiquassoException(
                f"An error occurred when evaluating the condition '{self._condition}' "
                f"for the instruction '{self}': {e}\n"
                f"Make sure that the condition is a callable that takes the "
                f"tuple of measurement outcomes and returns a boolean."
            ) from e


class BatchInstruction(Instruction):
    """Base class to control batch processing."""

    _execute: Callable


class Preparation(Instruction):
    """Base class for preparations."""


class Gate(Instruction):
    """Base class for gates."""


class Measurement(Instruction):
    r"""Base class for measurements."""
