import matplotlib.pyplot as plt
import json
import sys

# NOTE: https://pytest-benchmark.readthedocs.io/en/latest/comparing.html
#      Use --benchmark-autosave or --benchmark-save=<some-name> when running benchmarks.
#      Don't add extension, json is appended on default.

benchmark_file = open(sys.argv[1])

benchmark_json_file = json.load(benchmark_file)

piquasso_runs = []
strawberryfields_runs = []
# Separating data
for benchmark in benchmark_json_file["benchmarks"]:
    if "piquasso" in benchmark["name"]:
        piquasso_runs.append(benchmark)
    else:
        strawberryfields_runs.append(benchmark)

x_data = []
pq_y_data = []
sf_y_data = []

# Filtering data into variables to plot
for i in range(len(piquasso_runs)):
    # print(
    #     piquasso_runs[i]["param"], # for example: "3-interferometer0".
    #     piquasso_runs[i]["stats"]["mean"],
    #     strawberryfields_runs[i]["stats"]["mean"],
    #     )
    x_data.append(
        int(piquasso_runs[i]["param"][0])
    )  # Amount of modes, or cutoff possibly.
    pq_y_data.append(float(piquasso_runs[i]["stats"]["mean"]))
    sf_y_data.append(float(strawberryfields_runs[i]["stats"]["mean"]))

fig, ax = plt.subplots()

ax.plot(x_data, pq_y_data)
ax.plot(x_data, sf_y_data)

# Parameters which possibly need to be set depending on the benchmark
ax.set_title(piquasso_runs[0]["group"])
ax.set_xlabel("number of modes")
ax.set_ylabel("runtime (ms)")
ax.legend(["piquasso", "strawberryfields"])
ax.set_xticks(x_data)
ax.set_yscale("log")

plt.savefig(sys.argv[2])

benchmark_file.close()
