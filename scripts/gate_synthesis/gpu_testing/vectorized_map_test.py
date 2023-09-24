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
print("Mapping matrix multiplication test started")
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
print("Modules loaded")
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
sys_details = tf.sysconfig.get_build_info()
cuda = sys_details["cuda_version"]
cudnn = sys_details["cudnn_version"]
print("CUDA: {}; CUDNN: {}".format(cuda, cudnn))

#### General GPU test
matrix_size = 10
number_of_matrices = 1000
iterations_to_average_over = 20

print("map_fn on CPU--------------------------------------")
with tf.device("/CPU:0"):
    @tf.function(jit_compile=True)
    def my_matmul_cpu(matrices):
        return tf.matmul(matrices[0], matrices[1])
    start_time = time.time()
    for _ in range(iterations_to_average_over):
        matrices = tf.random.uniform((number_of_matrices, 2, matrix_size, matrix_size))
        tf.map_fn(my_matmul_cpu, matrices)

    duration = time.time() - start_time
    print("CPU map_fn sum_time:", duration)

print("map_fn on GPU--------------------------------------")
@tf.function(jit_compile=True)
def my_matmul(matrices):
    return tf.matmul(matrices[0], matrices[1])

start_time = time.time()
for _ in range(iterations_to_average_over):
    matrices = tf.random.uniform((number_of_matrices, 2, matrix_size, matrix_size))
    tf.map_fn(my_matmul, matrices)

duration = time.time() - start_time
print("GPU map_fn sum_time:", duration)

print("vectorized on GPU----------------------------------")
start_time = time.time()
for _ in range(iterations_to_average_over):
    matrices = tf.random.uniform((number_of_matrices, 2, matrix_size, matrix_size))
    tf.vectorized_map(my_matmul, matrices)

duration = time.time() - start_time
print("GPU map_fn sum_time:", duration)

print("vectorized on CPU----------------------------------")
with tf.device("/CPU:0"):
    @tf.function(jit_compile=True)
    def my_matmul_cpu(matrices):
        return tf.matmul(matrices[0], matrices[1])
    start_time = time.time()
    for _ in range(iterations_to_average_over):
        matrices = tf.random.uniform((number_of_matrices, 2, matrix_size, matrix_size))
        tf.vectorized_map(my_matmul_cpu, matrices)

    duration = time.time() - start_time
    print("CPU map_fn sum_time:", duration)
print("Finished")
exit()