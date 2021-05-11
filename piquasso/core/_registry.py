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


items = {}


def use_plugin(plugin, override=False):
    for name, class_ in plugin.classes.items():
        class_.__name__ = name
        if override or name not in items:
            items[name] = class_


def create_instance_from_mapping(mapping):
    """Creates an instance using the `registry` classes from a mapping.

    The supported mapping format is:

    .. code-block:: python

        {
            "type": <CLASS_NAME>,
            "properties": {
                ...
            }
        }

    The value under `"type"` will be searched in the `registry` for the corresponding
    class, and the `"properties"` will be used to initialize the class with.

    Args:
        mapping (collections.Mapping): The instance represented in a mapping.

    Returns:
        The created instance corresponding to the `mapping` specified.
    """

    class_ = items[mapping["type"]]

    return class_.from_properties(mapping["properties"])
