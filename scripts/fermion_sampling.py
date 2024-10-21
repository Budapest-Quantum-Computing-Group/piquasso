import numpy as np
from piquasso._math.combinatorics import is_even_permutation
import piquasso as pq
connector = pq.NumpyConnector()


def get_fermion_sampling_initial_tensor(N:int, l: int):
    tensor = np.empty(shape=(2*N,)*(2*l), dtype = complex)
    size = (2*N)**(2*l)
    for idx in range(size):
        tensor_index = np.empty(shape=2*l)
        for i in range(2*l):
            tensor_index[2*l-i-1] = idx % (2*N)
            idx = idx // (2*N)
        prefactor, filtered_tensor_index = get_prefactor(tensor_index)


def psi():
    tensor = np.empty(shape=(2,) * ( 2 * 4 ), dtype = complex)
    size=2**8
    for idx in range(size):
        tensor_index = np.empty(shape = 2 * 4, dtype=int)
        for i in range(2*4):
            tensor_index[2*4-i-1] = idx % (2*1)
            idx = idx // (2*1)
        first_quantized_index = np.arange(2*4)[tensor_index == 1]
        extended = np.empty(len(first_quantized_index)+4, dtype=first_quantized_index.dtype)
        extended[2:-2] = first_quantized_index

        I = np.identity(4)
        O = np.zeros_like(I)
        gamma = np.block([[O,I], [-I, O]])

        sum = 0.0
        for i in range(2):
            extended[-2:] = [2*i, 2*i+1]
            for j in range(2):
                extended[:2] = [2*j+1,2*j]
                prefactor, filtered_tensor_index = get_prefactor(extended)
                reduced_gamma = gamma[np.ix_(filtered_tensor_index, filtered_tensor_index)]
                sum += prefactor * connector.pfaffian(reduced_gamma)

        sum /= 2

        tensor[tensor_index] = sum
    return tensor
        
        
def get_prefactor(tensor_index):
    sorter = np.argsort(tensor_index)
    prefactor = 1 if is_even_permutation(sorter) else -1
    sorted_tensor_index = tensor_index[sorter]
    filtered_tensor_index = np.unique(sorted_tensor_index)
    prefactor /= 2**(len(sorted_tensor_index)-len(filtered_tensor_index))

    return prefactor, filtered_tensor_index
        
        
        

