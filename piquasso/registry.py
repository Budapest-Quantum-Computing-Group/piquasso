#
# Copyright (C) 2020 by TODO - All rights reserved.
#


class ClassRecorder:
    """Class to record class definitions throughout the package.

    To record a class, just subclass this class.

    Attributes:
        records (dict): A map of the classes to access them by their names.
    """

    records = {}

    def __init_subclass__(cls, **kwargs):
        """Adds the subclass to :attr:`records`."""
        super().__init_subclass__(**kwargs)

        cls.records[cls.__name__] = cls

    @classmethod
    def from_properties(cls, properties):
        """Creates an instance from a mapping specified.

        Args:
            properties (collections.Mapping):
                The desired instance in the format of a mapping.
        """

        raise NotImplementedError(
            f"No 'from_properties' classmethod is implemented for class '{cls}'."
        )


def set_class(name, class_):
    ClassRecorder.records[name] = class_
    class_.__name__ = name


def retrieve_class(class_name):
    class_ = ClassRecorder.records.get(class_name)

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
