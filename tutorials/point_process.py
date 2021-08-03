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

from piquasso._math.points import *

def quantum_inspired():
    size = 4
    R = np.array([(i, j) for i in range(size) for j in range(size)])
    K = rbf_kernel(R, 2.5)

    samples = quantum_inspired_points_sample(K, 50.0, 10)
    points(R, samples[0], point_size=10)

def ppp():
    size = 4
    R = np.array([(i, j) for i in range(size) for j in range(size)])
    K = rbf_kernel(R, 2.5)

    samples = permanent_points_sample(K, size, 50.0, 10)
    points(R, samples[0], point_size=10).show()


quantum_inspired()
