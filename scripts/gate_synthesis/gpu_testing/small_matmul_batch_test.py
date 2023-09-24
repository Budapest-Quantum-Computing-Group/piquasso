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
print("Small matrix multiplication test started")
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
matrix_size = 50
number_of_matrices = 10000
iterations_to_average_over = 100

print("GPU Data GPU Calculation---------------------------")
start_time = time.time()
for _ in range(iterations_to_average_over):
    mat_a = tf.random.uniform((number_of_matrices, matrix_size, matrix_size))
    mat_b = tf.random.uniform((number_of_matrices, matrix_size, matrix_size))
    tf.einsum("bij,bjk->bik", mat_a, mat_b)

duration = time.time() - start_time
print("GPU-GPU sum_time:", duration)

print("CPU Data CPU Calculation---------------------------")
with tf.device("/CPU:0"):
    start_time = time.time()
    for _ in range(iterations_to_average_over):
        mat_a = tf.random.uniform((number_of_matrices, matrix_size, matrix_size))
        mat_b = tf.random.uniform((number_of_matrices, matrix_size, matrix_size))
        tf.einsum("bij,bjk->bik", mat_a, mat_b)

    duration = time.time() - start_time
    print("CPU-CPU sum_time:", duration)

print("CPU Data GPU Calculation---------------------------")
start_time = time.time()
with tf.device("/CPU:0"):
    mat_a = tf.random.uniform((iterations_to_average_over, number_of_matrices, matrix_size, matrix_size))
    mat_b = tf.random.uniform((iterations_to_average_over, number_of_matrices, matrix_size, matrix_size))
for iteration in range(iterations_to_average_over):
    tf.einsum("abij,abjk->abik", mat_a, mat_b)

duration = time.time() - start_time
print("CPU-GPU sum_time:", duration)

print("GPU Data CPU Calculation---------------------------")
start_time = time.time()
for _ in range(iterations_to_average_over):
    mat_a = tf.random.uniform((number_of_matrices, matrix_size, matrix_size))
    mat_b = tf.random.uniform((number_of_matrices, matrix_size, matrix_size))
    with tf.device("/CPU:0"):
        tf.einsum("bij,bjk->bik", mat_a, mat_b)

duration = time.time() - start_time
print("GPU-CPU sum_time:", duration)

print("Script finished")
exit()