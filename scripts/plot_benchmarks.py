import matplotlib.pyplot as plt
import json

sf1_benchmark_file = open("./scripts/json_dump/pq/20230222-220749_cost-func_2-14_modes_complete_c5.json")
sf2_benchmark_file = open("./scripts/json_dump/pq/20230222-232506_cost-func_2-14_modes_complete_c6.json")
sf3_benchmark_file = open("./scripts/json_dump/pq/20230222-120848_cost-func_2-14_modes_complete_c7.json")

sf1_benchmark_json_file = json.load(sf1_benchmark_file)
sf2_benchmark_json_file = json.load(sf2_benchmark_file)
sf3_benchmark_json_file = json.load(sf3_benchmark_file)

x_data = []
# cutoff = sf1_benchmark_json_file["benchmarks"][0]["cutoff"]  # NOTE: Alternatively "mode"
sf1_y_data = []
sf2_y_data = []
sf3_y_data = []

# Separating data
for benchmark in sf1_benchmark_json_file["benchmarks"]:
    x_data.append(benchmark["mode"])  # NOTE: Alternatively "cutoff"
    sf1_y_data.append(benchmark["pq"]["mean_exec_time"] + benchmark["pq"]["mean_gradient_time"])

for benchmark in sf2_benchmark_json_file["benchmarks"]:
    sf2_y_data.append(benchmark["pq"]["mean_exec_time"] + benchmark["pq"]["mean_gradient_time"])

for benchmark in sf3_benchmark_json_file["benchmarks"]:
    sf3_y_data.append(benchmark["pq"]["mean_exec_time"] + benchmark["pq"]["mean_gradient_time"])

fig, ax = plt.subplots(figsize = (12, 10))

ax.plot(x_data[:len(sf1_benchmark_json_file["benchmarks"])], sf1_y_data, marker="x", markersize=12)
ax.plot(x_data[:len(sf2_benchmark_json_file["benchmarks"])], sf2_y_data, marker="o",  markersize=10)
ax.plot(x_data[:len(sf3_benchmark_json_file["benchmarks"])], sf3_y_data, marker="D",  markersize=10)

# Parameters which possibly need to be set depending on the benchmark
ax.set_xlabel("number of modes [-]", fontsize = 22)
ax.set_ylabel("calculation time [ms]", fontsize = 22)
ax.legend(["cutoff: 5", "cutoff: 6", "cutoff: 7"], loc=2, prop={'size': 18})
ax.set_xticks(x_data)
plt.tick_params(axis='both', which='major', labelsize=18, length=8, width=1.5)
plt.tick_params(axis='both', which='minor', labelsize=18, length=5, width=1.2)
ax.set_yscale("log")

dpi = 300
plt.savefig("./prof/plots/increase_qmode_ctoff_com_pq.png", dpi=dpi)

sf1_benchmark_file.close()
sf2_benchmark_file.close()
sf3_benchmark_file.close()
