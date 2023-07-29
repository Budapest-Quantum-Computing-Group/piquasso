import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
sys_details = tf.sysconfig.get_build_info()
cuda = sys_details["cuda_version"]
cudnn = sys_details["cudnn_version"]
print(cuda, cudnn)

gpu_start = time.time()

a_gpu = tf.random.uniform((15000, 15000))
b_gpu = tf.random.uniform((15000, 15000))
c_gpu = tf.matmul(a_gpu, b_gpu)
# print(c_gpu)
print(time.time() - gpu_start)
# cpu_start = time.time()
# with tf.device('/CPU:0'):
#    a_cpu =  tf.random.uniform((15000, 15000))
#    b_cpu =  tf.random.uniform((15000, 15000))
#    c_cpu = tf.matmul(a_cpu, b_cpu)
# print(c_cpu)
# print(time.time()-cpu_start)
