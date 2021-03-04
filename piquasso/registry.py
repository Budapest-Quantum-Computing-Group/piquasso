#
# Copyright (C) 2020 by TODO - All rights reserved.
#

"""Module to store class definitions."""


_items = {}


def add(class_):
    if class_.__name__ not in _items:
        _items[class_.__name__] = class_

    return class_


def use_plugin(plugin, override=False):
    for name, class_ in plugin.classes.items():
        class_.__name__ = name
        if override or name not in _items:
            _items[name] = class_


def retrieve_class(class_name):
    class_ = _items.get(class_name)

    if class_ is None:
        raise NameError(f"Class with name '{class_name}' not found in registry.")

    return class_


def create_instance_from_mapping(mapping):
    """Creates an instance using the `registry` classes from a mapping.

    The supported mapping format is:
    ```
    {
        "type": <CLASS_NAME>,
        "properties": {
            ...
        }
    }
    ```

    The value under `"type"` will be searched in the `registry` for the corresponding
    class, and the `"properties"` will be used to initialize the class with.

    Args:
        mapping (collections.Mapping): The instance represented in a mapping.

    Returns:
        The created instance corresponding to the `mapping` specified.
    """

    class_ = retrieve_class(mapping["type"])

    return class_.from_properties(mapping["properties"])
