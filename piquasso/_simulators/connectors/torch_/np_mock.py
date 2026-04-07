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

# This module contains mock-ups used by the TorchConnector.

import torch


def sum(tensor):
    return torch.sum(tensor)

def real(tensor):
    return torch.real(tensor)

def array(input):
    # NOTE(TR): Does it make sense? Seems a little misleading, but the intuition seems correct.
    return torch.Tensor(input)

def diag(tensor):
    return torch.diag(tensor)

def zeros(size):
    return torch.zeros(size)

def conj(input):
    return torch.conj(input)

def allclose(input, other, rtol = 1e-05, atol = 1e-08, equal_nan = False):
    return torch.allclose(input, other, rtol, atol, equal_nan)

def copy(input):
    return input.detach().clone()

def abs(input):
    return torch.abs(input)

def outer(v1, v2):
    return torch.outer(v1, v2)