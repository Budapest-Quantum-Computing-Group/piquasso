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

from piquasso.core.mixins import _PropertyMixin, _RegisterMixin, _CodeMixin


class Instruction(_PropertyMixin, _RegisterMixin, _CodeMixin):
    """
    Args:
        *params: Variable length argument list.
    """

    def __init__(self, **params):
        self._set_params(**params)

    def _set_params(self, **params):
        self.params = params

    def _as_code(self):
        if hasattr(self, "modes"):
            mode_string = ", ".join([str(mode) for mode in self.modes])
        else:
            mode_string = ""

        if hasattr(self, "params"):
            params_string = "{}".format(
                ", ".join(
                    [f"{key}={value}" for key, value in self.params.items()]
                )
            )
        else:
            params_string = ""

        return f"pq.Q({mode_string}) | pq.{self.__class__.__name__}({params_string})"

    def on_modes(self, *modes):
        self.modes = modes
        return self

    def _apply_to_program_on_register(self, program, register):
        program.instructions.append(self.on_modes(*register.modes))

    @classmethod
    def from_properties(cls, properties: dict):
        """Creates an :class:`Instruction` instance from a mapping specified.

        Args:
            properties (dict):
                The desired :class:`Instruction` instance in
                the format of a mapping.

        Returns:
            Instruction:
                An :class:`Instruction` initialized using the
                specified mapping.
        """

        instruction = cls(**properties["params"])

        instruction.modes = properties["modes"]

        return instruction

    def __repr__(self):
        if hasattr(self, "modes"):
            modes = "modes={}".format(self.modes)
        else:
            modes = ""

        if hasattr(self, "params"):
            params = "{}".format(
                ", ".join(
                    [f"{key}={value}" for key, value in self.params.items()]
                )
            )
        else:
            params = ""

        classname = self.__class__.__name__

        return f"<pq.{classname}({params}, {modes})>"

    def __eq__(self, other):
        return (
            self.modes == other.modes
            and
            self.params == other.params
        )
