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
print("Large matrix multiplication test started")
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

#### Large matrix multiplication test
max_matrix_size = 16000
step_size = 1000
iterations_to_average_over = 20

matrix_sizes = range(step_size,max_matrix_size,step_size)
jit_step_amount = 10
max_matrix_size = 16000
step_size = 1000
@tf.function(jit_compile=True)
def matmul_jit_benchmark(float_param=0.5):
    sum_time = 0.0
    tf_matrix_sizes = tf.range(step_size,max_matrix_size,step_size)

    a_matrices = [tf.random.uniform((tf_matrix_sizes[i], tf_matrix_sizes[i])) for i in range(jit_step_amount)]
    b_matrices = [tf.random.uniform((tf_matrix_sizes[i], tf_matrix_sizes[i])) for i in range(jit_step_amount)]

    for i in range(jit_step_amount):
        start_time = time.time()
        for _ in tf.range(iterations_to_average_over):
            mat_c = float_param * tf.matmul(a_matrices[i], b_matrices[i])
            print("shape: {}, float_param: {}".format(mat_c.shape, float_param))
        duration = time.time() - start_time
        print(duration)
        sum_time += duration

    return float_param


print("JIT GPU-----------------------")
float_param = matmul_jit_benchmark()
print("First JIT float_param:", float_param)
breakpoint()
value_tensor = tf.constant(0.1)
value_tensor = tf.range(0, 1.0, 0.1)
for i in range(10):
    float_param = matmul_jit_benchmark(value_tensor)
    print("JIT float_param:", float_param)

print("GPU---------------------------")
sum_time = 0.0
for matrix_size in matrix_sizes:
    mat_a = tf.random.uniform((matrix_size, matrix_size))
    mat_b = tf.random.uniform((matrix_size, matrix_size))
    start_time = time.time()
    for _ in range(iterations_to_average_over):
        tf.matmul(mat_a, mat_b)
    end_time = time.time()
    duration = end_time-start_time
    sum_time += duration
    print("Matmul finished in {} over {} x {} matrix".format(
        duration,
        matrix_size,
        matrix_size
        ))
print("GPU sum_time:", sum_time)


print("CPU---------------------------")
with tf.device("/CPU:0"):
    sum_time = 0.0
    for matrix_size in matrix_sizes:
        mat_a = tf.random.uniform((matrix_size, matrix_size))
        mat_b = tf.random.uniform((matrix_size, matrix_size))
        start_time = time.time()
        for i in range(iterations_to_average_over):
            tf.matmul(mat_a, mat_b)
        end_time = time.time()
        duration = end_time-start_time
        print("Matmul finished in {} over {} x {}.".format(
            duration,
            matrix_size,
            matrix_size
            ))

print("Script finished")
exit()