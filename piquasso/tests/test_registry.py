#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

from unittest.mock import Mock

from piquasso import registry


def test_ClassRecorder_stores_subclasses():
    class SomeClass(registry.ClassRecorder):
        pass

    class SomeSubClass(SomeClass):
        pass

    class SomeSubSubClass(SomeSubClass):
        pass

    assert registry.retrieve_class("SomeSubClass") == SomeSubClass
    assert registry.retrieve_class("SomeSubSubClass") == SomeSubSubClass


def test_retrieving_nonexistent_class_raises_NameError():
    registry.ClassRecorder.records = {
        "existing_class": Mock(),
        "other_existing_class": Mock(),
    }

    with pytest.raises(NameError):
        registry.retrieve_class("nonexistent_class")
