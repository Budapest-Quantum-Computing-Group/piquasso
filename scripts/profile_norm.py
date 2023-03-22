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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import strawberryfields as sf
from strawberryfields import ops
import piquasso as pq
from piquasso._math.fock import cutoff_cardinality
import time
import numpy as np
import json


def pq_benchmark():
    print("PIQUASSO_START")
    def _get_number_of_beamsplitters(d: int):
        number_of_beamsplitters: int
        if d % 2 == 0:
            number_of_beamsplitters = (d // 2) ** 2
            number_of_beamsplitters += ((d - 1) // 2) * (d // 2)
        else:
            number_of_beamsplitters = ((d - 1) // 2) * d

        return number_of_beamsplitters


    def create_layer_parameters(d: int, number_of_layers: int,):
        number_of_beamsplitters = _get_number_of_beamsplitters(d)

        dtype = np.float64

        thetas_1 = np.random.uniform(size=(number_of_layers, number_of_beamsplitters), high=np.pi*2)
        phis_1 = np.random.uniform(size=(number_of_layers, d-1), high=np.pi*2)

        squeezings = np.ones(shape=[number_of_layers, d])*0.1

        thetas_2 = np.random.uniform(size=(number_of_layers, number_of_beamsplitters), high=np.pi*2)
        phis_2 = np.random.uniform(size=(number_of_layers, d-1), high=np.pi*2)

        displacements_r = np.ones(shape=[number_of_layers, d])*0.1
        displacements_phi = np.zeros(shape=[number_of_layers, d])

        kappas = np.random.uniform(size=(number_of_layers, d))

        weights = np.concatenate(
            [thetas_1, phis_1, squeezings, thetas_2, phis_2, displacements_r, displacements_phi, kappas], axis=1
        )

        return weights


    d = 7
    cutoff = 14
    number_of_layers = 8
    min_d = 3
    max_d = d
    min_cutoff = min_d
    max_cutoff = cutoff
    min_number_of_layers = 1
    max_number_of_layers = number_of_layers

    benchmark_json_list = {'benchmarks': []}
    file_name = "./scripts/benchmark_results/pq/photon_losing/{}.json".format(time.strftime("%Y%m%d-%H%M%S"))
    for number_of_layers in range(min_number_of_layers, max_number_of_layers):
        for d in range(min_d, max_d):
            for cutoff in range(min_cutoff, max_cutoff):
                print("layernum:{}_mode:{}_cutoff:{}".format(number_of_layers, d, cutoff))

                simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

                parameters = create_layer_parameters(d, number_of_layers)

                def create_interferometer(thetas, phis):
                    with pq.Program() as interferometer:
                        i = 0
                        for col in range(d):
                            if col % 2 == 0:
                                for mode in range(0, d - 1, 2):
                                    modes = (mode, mode + 1)
                                    pq.Q(*modes) | pq.Beamsplitter(thetas[i], phi=0.0)
                                    i += 1

                            if col % 2 == 1:
                                for mode in range(1, d - 1, 2):
                                    modes = (mode, mode + 1)
                                    pq.Q(*modes) | pq.Beamsplitter(thetas[i], phi=0.0)
                                    i += 1

                        for i in range(d - 1):
                            pq.Q(i) | pq.Phaseshifter(phis[i])

                    return interferometer

                def create_layer(single_layer_parameters):
                    k = _get_number_of_beamsplitters(d)

                    thetas_1 = single_layer_parameters[:k]
                    phis_1 = single_layer_parameters[k:k+d-1]

                    squeezings = single_layer_parameters[k+d-1: k+2*d-1]

                    thetas_2 = single_layer_parameters[k+2*d-1: 2*k+2*d-1]
                    phis_2 = single_layer_parameters[2*k+2*d-1: 2*k+3*d-2]

                    displacements_r = single_layer_parameters[2*k+3*d-2: 2*k+4*d-2]
                    displacements_phi = single_layer_parameters[2*k+4*d-2: 2*k+5*d-2]

                    kappas = single_layer_parameters[2*k+5*d-2: 2*k+6*d-2]

                    first_interferometer = create_interferometer(thetas_1, phis_1)
                    second_interferometer = create_interferometer(thetas_2, phis_2)

                    with pq.Program() as single_layer:
                        pq.Q(all) | first_interferometer

                        pq.Q(all) | pq.Squeezing(squeezings)

                        pq.Q(all) | second_interferometer

                        for i in range(d - 1):
                            pq.Q(i) | pq.Phaseshifter(phis_2[i])

                        pq.Q(all) | pq.Displacement(r=displacements_r, phi=displacements_phi)
                        pq.Q(all) | pq.Kerr(kappas)

                    return single_layer

                layers = [create_layer(parameters[i]) for i in range(parameters.shape[0])]

                with pq.Program() as program:
                    pq.Q(all) | pq.Vacuum()

                    for layer in layers:
                        pq.Q(all) | layer

                start_time = time.time()
                state = simulator.execute(program).state
                exec_time = time.time() - start_time
                norms = state.norm_values
                result = 1
                for norm in norms:
                    result *= float(norm)

                benchmark_values = {'mode':d,
                                    'cutoff': cutoff,
                                    'number_of_layers': number_of_layers,
                                    'photons_lost': 1.0 - result,
                                    'exec_time': exec_time,
                                    }

                benchmark_json_list['benchmarks'].append(benchmark_values)

                with open(file_name, "w") as file:
                    json.dump(benchmark_json_list, file, indent=6)


def sf_benchmark():
    print("STRAWBERRYFIELDS_START")
    """
    This code has been copied from the following website with minor modifications:
    https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html
    """

    def interferometer(params, q):
        N = len(q)
        theta = params[: N * (N - 1) // 2]
        phi = params[N * (N - 1) // 2 : N * (N - 1)]
        rphi = params[-N + 1 :]

        if N == 1:
            ops.Rgate(rphi[0]) | q[0]
            return

        n = 0

        for i in range(N):
            for j, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (i + j) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1

        for i in range(max(1, N - 1)):
            ops.Rgate(rphi[i]) | q[i]


    def layer(params, q):
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)

        int1 = params[:M]
        s = params[M : M + N]
        int2 = params[M + N : 2 * M + N]
        dr = params[2 * M + N : 2 * M + 2 * N]
        dp = params[2 * M + 2 * N : 2 * M + 3 * N]
        k = params[2 * M + 3 * N : 2 * M + 4 * N]

        # begin layer
        interferometer(int1, q)

        for i in range(N):
            ops.Sgate(s[i]) | q[i]

        interferometer(int2, q)

        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]
            ops.Kgate(k[i]) | q[i]


    def init_weights(modes, layers):
        M = int(modes * (modes - 1)) + max(1, modes - 1)

        int1_weights = np.random.uniform(size=(layers, M), high=np.pi*2)
        s_weights = np.ones(shape=[layers, modes])*0.1
        int2_weights = np.random.uniform(size=(layers, M), high=np.pi*2)
        dr_weights = np.ones(shape=[layers, modes])*0.1
        dp_weights = np.zeros(shape=[layers, modes])
        k_weights = np.random.uniform(size=(layers, modes))

        weights = np.concatenate(
            [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights],
            axis=1,
        )

        return weights


    np.random.seed(137)

    d = 7
    cutoff = 14
    number_of_layers = 8
    min_d = 3
    max_d = d
    min_cutoff = min_d
    max_cutoff = cutoff
    min_number_of_layers = 1
    max_number_of_layers = number_of_layers

    benchmark_json_list = {'benchmarks': []}
    file_name = "./scripts/benchmark_results/sf/photon_losing/{}.json".format(time.strftime("%Y%m%d-%H%M%S"))
    for number_of_layers in range(min_number_of_layers, max_number_of_layers):
        for d in range(min_d, max_d):
            for cutoff in range(min_cutoff, max_cutoff):
                print("layernum:{}_mode:{}_cutoff:{}".format(number_of_layers, d, cutoff))

                eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
                qnn = sf.Program(d)

                weights = init_weights(d, number_of_layers)
                num_params = np.prod(weights.shape)

                sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
                sf_params = np.array([qnn.params(*i) for i in sf_params])

                with qnn.context as q:
                    for i in range(d):
                        sf.ops.Dgate(0.1) | q[i]

                    for k in range(number_of_layers):
                        layer(sf_params[k], q)

                mapping = {
                    p.name: w for p, w in zip(sf_params.flatten(), np.reshape(weights, [-1]))
                }

                start_time = time.time()
                state = eng.run(qnn, args=mapping).state
                exec_time = time.time() - start_time

                ket = state.ket()
                flatten_ket = np.reshape(ket, (-1))
                norm = np.real(np.dot(flatten_ket, np.conj(flatten_ket)))

                benchmark_values = {'mode':d,
                                    'cutoff': cutoff,
                                    'number_of_layers': number_of_layers,
                                    'photons_lost': 1.0 - norm,
                                    'exec_time': exec_time,
                                    }

                benchmark_json_list['benchmarks'].append(benchmark_values)

                with open(file_name, "w") as file:
                    json.dump(benchmark_json_list, file, indent=6)

pq_benchmark()
sf_benchmark()