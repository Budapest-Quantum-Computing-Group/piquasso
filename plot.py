import json

pq_results = open("pq_results.json")
sf_results = open("sf_results.json")

pq_data = json.load(pq_results)["benchmarks"]
sf_data = json.load(sf_results)["benchmarks"]

pq_results.close()
sf_results.close()

pq_losses = []
pq_exec_times = []

for point in pq_data:
    pq_losses.append(point["photons_lost"])
    pq_exec_times.append(point["exec_time"])


sf_losses = []
sf_exec_times = []

for point in sf_data:
    sf_losses.append(point["photons_lost"])
    sf_exec_times.append(point["exec_time"])


import matplotlib.pyplot as plt


plt.scatter(pq_losses, pq_exec_times, c='b', marker='x', label='Piquasso')
plt.scatter(sf_losses, sf_exec_times, c='r', marker='s', label='Strawberry Fields')

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Photon loss [-]")
plt.ylabel("Execution time [s]")

plt.legend(loc='lower left')
plt.show()

