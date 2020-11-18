#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from . import registry


class State(registry.ClassRecorder):
    @classmethod
    def from_properties(cls, properties):
        return cls(**properties)
