#
# Copyright (C) 2020 by TODO - All rights reserved.
#


class ClassRecorder:
    """Class to record class definitions throughout the package.

    To record a class, just subclass this class.
    """
    records = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.records[cls.__name__] = cls


def retrieve_class(class_name):
    class_ = ClassRecorder.records.get(class_name)

    if class_ is None:
        raise NameError(f"Class with name '{class_name}' not found in registry.")

    return class_
