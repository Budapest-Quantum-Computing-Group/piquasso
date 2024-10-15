import numpy as np
from piquasso._math.combinatorics import is_even_permutation


def get_fermion_sampling_initial_tensor(N:int, l: int):
    tensor = np.empty(shape=(2*N,)*(2*l), dtype = complex)
    size = (2*N)**(2*l)
    for idx in range(size):
        tensor_index = np.empty(shape=2*l)
        for i in range(2*l):
            tensor_index[2*l-i-1] = idx % (2*N)
            idx = idx // (2*N)
        sorter = np.argsort(tensor_index)
        prefactor = 1 if is_even_permutation(sorter) else -1
        sorted_tensor_index = tensor_index[sorter]

        
        

