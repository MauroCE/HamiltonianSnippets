import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font', **{'family': 'STIXGeneral'})

with open(
        "../results_storage/adaptT_Tmax80_T80_seed5095905818_N500_T80_massFalse_runs20_from0dot001_to10dot0_skewness3.pkl", "rb") as file:
    results = pickle.load(file)

n_runs = 20
n_eps = 9

initial_epsilons = np.geomspace(start=0.001, stop=10.0, num=9)
logLts = np.array([res['logLt'] for res in results]).reshape(n_runs, n_eps)
final_eps_means = np.array([res['out']['epsilon_params_history'][-1]['mean'] for res in results]).reshape(n_runs, n_eps)

colors = ['lightcoral', 'darkorange', 'gold',
          'olivedrab', 'lightseagreen', 'cornflowerblue',
          'mediumorchid', 'darkgrey', 'darkgreen']

fig, ax = plt.subplots()
for i in range(n_runs):
    for j in range(n_eps):
        r = results[i*n_eps + j]['out']
        label = r"$\mathregular{" + str(np.round(initial_epsilons[j], 3)) + "}$" if i == 0 else None
        ax.plot(r['gammas'], r['T_history'], c=colors[j], alpha=0.5, label=label)
# ax.set_xscale('log')
ax.set_xlabel(r"$\mathregular{\gamma}$", fontsize=13)
ax.set_ylabel(r"$\mathregular{T}$", fontsize=13)
ax.grid(True, color='gainsboro')
ax.legend(title=r"$\mathregular{\epsilon}$")
plt.show()


gap = 0.4
positions = np.arange(0, n_eps)
show_fliers = True
plot_logLt = False
values_to_plot = logLts if plot_logLt else final_eps_means
y_label = "Log Normalizing Constant" if plot_logLt else "Final Epsilon Mean"
fig, ax = plt.subplots(figsize=(15, 4))
bp_adapt = ax.boxplot(x=values_to_plot, showfliers=show_fliers, positions=positions, patch_artist=True,
                      boxprops=dict(facecolor="lightseagreen", color="k"),
                      capprops=dict(color="k"),
                      whiskerprops=dict(color="k"),
                      flierprops=dict(markerfacecolor="paleturquoise", markeredgecolor="k", marker='o'),
                      medianprops=dict(color="k"))
ax.set_xticks(positions)
ax.set_xticklabels(["{:.4f}".format(eps) for eps in initial_epsilons], rotation=0)
ax.set_xlabel("Initial Epsilons", fontsize=13)
ax.set_ylabel(y_label, fontsize=13)
ax.legend(handles=[bp_adapt["boxes"][0]], labels=["Hamiltonian Snippets"], loc="lower center", fontsize=9)
plt.show()
