import matplotlib.pyplot as plt
import json
import sys

if(len(sys.argv) <= 2):
    sf_benchmark_file = open("./scripts/json_dump/sf/20230221-153305_mean-position_2-5_modes_complete.json")
    pq_benchmark_file = open("./scripts/json_dump/pq/20230221-153305_mean-position_2-5_modes_complete.json")
else:
    sf_benchmark_file = open(sys.argv[2])
    pq_benchmark_file = open(sys.arv[3])

sf_benchmark_json_file = json.load(sf_benchmark_file)
pq_benchmark_json_file = json.load(pq_benchmark_file)

x_data = []
cutoff = sf_benchmark_json_file["benchmarks"][0]["cutoff"]  # NOTE: Alternatively "mode"
pq_y_data = []
sf_y_data = []

# Separating data
for benchmark in sf_benchmark_json_file["benchmarks"]:
    x_data.append(benchmark["mode"])  # NOTE: Alternatively "cutoff"
    sf_y_data.append(benchmark["sf"]["mean_exec_time"] + benchmark["sf"]["mean_gradient_time"])

for benchmark in pq_benchmark_json_file["benchmarks"]:
    pq_y_data.append(benchmark["pq"]["mean_exec_time"] + benchmark["pq"]["mean_gradient_time"])

fig, ax = plt.subplots(figsize = (12, 8))

ax.plot(x_data, pq_y_data, linewidth=3)
ax.plot(x_data, sf_y_data, linewidth=3)

# Parameters which possibly need to be set depending on the benchmark
ax.set_title("Mean position value", fontsize = 22)
ax.set_xlabel("number of modes", fontsize = 22)
ax.set_ylabel("runtime (ms)", fontsize = 22)
ax.legend(["piquasso", "strawberryfields"], loc=2, prop={'size': 18})
ax.set_xticks(x_data)
plt.tick_params(axis='both', which='major', labelsize=18)
ax.set_yscale("log")

dpi = 300
if len(sys.argv) == 1:
    plt.savefig("./prof/plots/increase_qmode.png", dpi=dpi)
else:
    plt.savefig(sys.argv[1], dpi=dpi)

sf_benchmark_file.close()
pq_benchmark_file.close()
