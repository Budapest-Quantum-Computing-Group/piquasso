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


class PiquassoException(Exception):
    """Base class for all exceptions raised by Piquasso."""


class InvalidState(PiquassoException):
    """Raised when an invalid state is encountered or being prepared."""


class InvalidParameter(PiquassoException):
    """Raised when an invalid parameter is specified."""


class InvalidModes(PiquassoException):
    """Raised when invalid set of modes are encountered."""


class InvalidProgram(PiquassoException):
    """Raised when an invalid program is being created or used."""


class InvalidInstruction(PiquassoException):
    """Raised when an invalid instruction is specified."""


class InvalidSimulation(PiquassoException):
    """Raised when a simulation could not be executed."""
