import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pickle
rc('font', **{'family': 'STIXGeneral'})

# Load results, this is a list of dictionaries
with open("results/seed1841913736_N500_T30_massFalse_runs20_from0dot001_to10dot0_skewness3.pkl", "rb") as file:
    results = pickle.load(file)

n_runs = 20
n_eps = 9

# Log normalizing constants and initial epsilons
initial_epsilons = np.array(np.geomspace(start=0.001, stop=10.0, num=n_eps))  # np.array() cmd used only for pylint
logLts = np.array([res['logLt'] for res in results]).reshape(n_runs, n_eps)

# Plot parameters
gap = 0.4
positions = np.arange(0, n_eps)
show_fliers = True

# Boxplots
fig, ax = plt.subplots(figsize=(15, 4))
bp_adapt = ax.boxplot(x=logLts, showfliers=show_fliers, positions=positions, patch_artist=True,
                      boxprops=dict(facecolor="lightseagreen", color="k"),
                      capprops=dict(color="k"),
                      whiskerprops=dict(color="k"),
                      flierprops=dict(markerfacecolor="paleturquoise", markeredgecolor="k", marker='o'),
                      medianprops=dict(color="k"))
ax.set_xticks(positions)
ax.set_xticklabels(["{:.4f}".format(eps) for eps in initial_epsilons], rotation=0)
ax.set_xlabel("Initial Epsilons", fontsize=13)
ax.set_ylabel("Log Normalizing Constant", fontsize=13)
ax.legend(handles=[bp_adapt["boxes"][0]], labels=["Hamiltonian Snippets"], loc="lower center", fontsize=9)
plt.savefig("boxplots_log_normalizing_constants.png")
plt.show()
