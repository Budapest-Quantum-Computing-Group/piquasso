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

"""Module to store class definitions."""
from typing import Type, Any

from piquasso.api.plugin import Plugin

items = {}


def use_plugin(plugin: Type[Plugin], override: bool = False) -> None:
    for name, class_ in plugin.classes.items():
        class_.__name__ = name
        if override or name not in items:
            items[name] = class_


def get_class(name: str) -> Any:
    return items[name]
