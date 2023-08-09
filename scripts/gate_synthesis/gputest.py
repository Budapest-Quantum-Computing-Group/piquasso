#
# Copyright 2021-2023 Budapest Quantum Computing Group
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

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
sys_details = tf.sysconfig.get_build_info()
cuda = sys_details["cuda_version"]
cudnn = sys_details["cudnn_version"]
print(cuda, cudnn)


a_gpu = tf.random.uniform((1500, 1500))
b_gpu = tf.random.uniform((1500, 1500))
gpu_start = time.time()
c_gpu = tf.matmul(a_gpu, b_gpu)
# print(c_gpu)
print(time.time() - gpu_start)
with tf.device("/CPU:0"):
    a_cpu = tf.random.uniform((1500, 1500))
    b_cpu = tf.random.uniform((1500, 1500))
    cpu_start = time.time()
    c_cpu = tf.matmul(a_cpu, b_cpu)
    print(time.time() - cpu_start)
