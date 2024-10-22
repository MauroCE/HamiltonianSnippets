import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font', **{'family': 'STIXGeneral'})

with open(
        "../results_storage/new_adaptT_sonar_seed5216451320_N500_T80_massFalse_runs20_from0dot001_to10dot0_skewness3_aooFalse_skipoFalse_minT2.pkl", "rb") as file:
    results = pickle.load(file)

# edit results since one of them was none
for i in range(len(results)):
    if not isinstance(results[i]['out']['T_history'], list) and isinstance(results[i]['out']['T_history'], float):
        results[i]['out']['T_history'] = [np.nan]
        results[i]['out']['epsilon_params_history'] = [{'mean': np.nan}]

n_runs = 20
n_eps = 9

initial_epsilons = np.geomspace(start=0.001, stop=10.0, num=9)
final_Ts = np.array([res['out']['T_history'][-1] for res in results]).reshape(n_runs, n_eps)
final_eps_means = np.array([res['out']['epsilon_params_history'][-1]['mean'] for res in results]).reshape(n_runs, n_eps)
final_taus = final_Ts * final_eps_means

final_taus_list = [final_taus[:, i][~np.isnan(final_taus[:, i])] for i in range(n_eps)]

colors = ['lightcoral', 'darkorange', 'gold',
          'olivedrab', 'lightseagreen', 'cornflowerblue',
          'mediumorchid', 'darkgrey', 'darkgreen']


# Plot parameters
gap = 0.4
positions = np.arange(0, n_eps)
show_fliers = True

# Boxplots for taus
# fig, ax = plt.subplots(figsize=(15, 4))
# bp_adapt = ax.boxplot(x=final_taus_list, showfliers=show_fliers, positions=positions, patch_artist=True,
#                       boxprops=dict(facecolor="lightseagreen", color="k"),
#                       capprops=dict(color="k"),
#                       whiskerprops=dict(color="k"),
#                       flierprops=dict(markerfacecolor="paleturquoise", markeredgecolor="k", marker='o'),
#                       medianprops=dict(color="k"))
# ax.set_xticks(positions)
# ax.set_xticklabels(["{:.4f}".format(eps) for eps in initial_epsilons], rotation=0)
# ax.set_xlabel(r"$\mathregular{\theta_0}$" + " (Initial Epsilons)", fontsize=13)
# ax.set_ylabel(r"$\mathregular{\tau_P = T_P \theta_P}$", fontsize=13)
# ax.legend(handles=[bp_adapt["boxes"][0]], labels=["Hamiltonian Snippets"], loc="lower center", fontsize=9)
# plt.savefig("boxplots_final_taus_N500_T80_skewness3_runs20.png")
# plt.show()


# Plot the T evolutions
log_scale = True
fig, ax = plt.subplots()
for i in range(n_runs):
    for eps_ix, eps in enumerate(initial_epsilons):
        index = i*n_eps + eps_ix
        gammas = results[index]['out']['gammas']
        T_history = results[index]['out']['T_history']
        if len(T_history) != 1:
            label = f"{eps: .3f}" if i == n_runs-1 else None
            ax.plot(gammas, T_history, color=colors[eps_ix], label=label)
if log_scale:
    ax.set_xscale('log')
ax.set_xlabel(r"$\mathregular{\gamma}$", fontsize=13)
ax.set_ylabel(r"$\mathregular{T}$", fontsize=13)
ax.grid(True, color='gainsboro')
ax.legend(title=r'$\mathregular{\epsilon}$', fontsize=11)
title = "evolution_of_T.png" if not log_scale else "evolution_of_T_log_scale.png"
plt.savefig(title)
plt.show()



# fig, ax = plt.subplots()
# for i in range(n_runs):
#     for j in range(n_eps):
#         r = results[i*n_eps + j]['out']
#         label = r"$\mathregular{" + str(np.round(initial_epsilons[j], 3)) + "}$" if i == 0 else None
#         ax.plot(r['gammas'], r['T_history'], c=colors[j], alpha=0.5, label=label)
# # ax.set_xscale('log')
# ax.set_xlabel(r"$\mathregular{\gamma}$", fontsize=13)
# ax.set_ylabel(r"$\mathregular{T}$", fontsize=13)
# ax.grid(True, color='gainsboro')
# ax.legend(title=r"$\mathregular{\epsilon}$")
# plt.show()


# gap = 0.4
# positions = np.arange(0, n_eps)
# show_fliers = True
# plot_logLt = False
# values_to_plot = logLts if plot_logLt else final_eps_means
# y_label = "Log Normalizing Constant" if plot_logLt else "Final Epsilon Mean"
# fig, ax = plt.subplots(figsize=(15, 4))
# bp_adapt = ax.boxplot(x=values_to_plot, showfliers=show_fliers, positions=positions, patch_artist=True,
#                       boxprops=dict(facecolor="lightseagreen", color="k"),
#                       capprops=dict(color="k"),
#                       whiskerprops=dict(color="k"),
#                       flierprops=dict(markerfacecolor="paleturquoise", markeredgecolor="k", marker='o'),
#                       medianprops=dict(color="k"))
# ax.set_xticks(positions)
# ax.set_xticklabels(["{:.4f}".format(eps) for eps in initial_epsilons], rotation=0)
# ax.set_xlabel("Initial Epsilons", fontsize=13)
# ax.set_ylabel(y_label, fontsize=13)
# ax.legend(handles=[bp_adapt["boxes"][0]], labels=["Hamiltonian Snippets"], loc="lower center", fontsize=9)
# plt.show()
