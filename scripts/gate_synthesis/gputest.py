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
print("Script started")
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
print("Modules imported")
m_size = 100

"""
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
sys_details = tf.sysconfig.get_build_info()
cuda = sys_details["cuda_version"]
cudnn = sys_details["cudnn_version"]
print(cuda, cudnn)

#### General GPU test

a_gpu = tf.random.uniform((m_size, m_size))
b_gpu = tf.random.uniform((m_size, m_size))
gpu_start = time.time()
c_gpu = tf.matmul(a_gpu, b_gpu)
# print(c_gpu)
print(time.time() - gpu_start)
with tf.device("/CPU:0"):
    a_cpu = tf.random.uniform((m_size, m_size))
    b_cpu = tf.random.uniform((m_size, m_size))
    cpu_start = time.time()
    c_cpu = tf.matmul(a_cpu, b_cpu)
    print(time.time() - cpu_start)

"""
#### Vectorized map test

matrix_amount = 400

@tf.function
def my_matrix_mul(mats):
    mat1 = mats[0]
    mat2 = mats[1]
    return tf.matmul(mat1, mat2)

@tf.function
def my_conjugate(mat):
    return tf.math.conj(mat)

print("Generating matrices")

matrices1 = tf.convert_to_tensor([tf.random.uniform((m_size, m_size))\
            for _ in range(matrix_amount)])

print("Matrix generation finished")

for i in range(200,matrix_amount,200):
    print("matrix_amount:", i)

    basic_time = time.time()
    for j in range(i):
        my_conjugate(matrices1[j])

    print("Basic time:", time.time() - basic_time)

    map_fn_time = time.time()

    # tf.map_fn(my_conjugate, matrices1)

    print("map_fn time:", time.time() - map_fn_time)

    vectmap_time = time.time()

    tf.vectorized_map(my_conjugate, matrices1)

    print("Vectorized time:", time.time() - vectmap_time)