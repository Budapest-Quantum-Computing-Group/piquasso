#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

from typing import Optional, List, Type

from piquasso.api.config import Config
from piquasso.api.connector import BaseConnector

from piquasso.api.simulator import Simulator
from piquasso.api.exceptions import InvalidSimulation

from .connectors.connector import BuiltinConnector


class BuiltinSimulator(Simulator):
    _extra_builtin_connectors: List[Type[BaseConnector]] = []

    def __init__(
        self,
        d: Optional[int] = None,
        config: Optional[Config] = None,
        connector: Optional[BaseConnector] = None,
    ) -> None:
        if connector is not None:
            self._validate_connector(connector)

        super().__init__(d=d, config=config, connector=connector)

    @classmethod
    def _validate_connector(cls, connector):
        if isinstance(connector, BuiltinConnector) and not isinstance(
            connector, cls._supported_connector_classes()
        ):
            raise InvalidSimulation(
                f"The connector '{connector}' is not supported."
                f"Supported connectors:"
                "\n"
                f"{cls._supported_connector_classes()}"
            )

    @classmethod
    def _supported_connector_classes(cls):
        return tuple([cls._default_connector_class] + cls._extra_builtin_connectors)
