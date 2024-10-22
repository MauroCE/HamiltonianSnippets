import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pickle
rc('font', **{'family': 'STIXGeneral'})

# Load results, this is a list of dictionaries
with open("../results_storage/newrun_seed4931862540_N500_T30_runs20_from0dot001_to10dot0_skewness3.pkl", "rb") as file:
    results_adapt = pickle.load(file)

with open(
        "../../../../IntegratorSnippetsExperiments_FromWorkPC/results/abg/results_N500_T30_from0dot001_to10_runs20.pkl", "rb") as file:
    results_fixed = pickle.load(file)

n_runs = 20
n_eps = 9

# Log normalizing constants and initial epsilons
initial_epsilons = np.array(np.geomspace(start=0.001, stop=10.0, num=n_eps))  # np.array() cmd used only for pylint
logLts_adapt = np.array([res['logLt'] for res in results_adapt]).reshape(n_runs, n_eps)
logLts_fixed = np.array([res['logLt'] for res in results_fixed]).reshape(n_runs, n_eps)

# Plot parameters
gap = 0.4
positions_start = 0
positions_every = 2
positions = np.arange(positions_start, positions_every*n_eps, positions_every)
adapt_positions = positions - gap
fixed_positions = positions + gap
vertical_lines_start = (positions_every - positions_start) / 2
vertical_lines = np.arange(vertical_lines_start, (n_eps-1)*positions_every, positions_every)
show_fliers = True

# Boxplots
# fig, ax = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)
# bp_adapt = ax[0].boxplot(x=logLts_adapt, showfliers=show_fliers, positions=adapt_positions, patch_artist=True,
#                          boxprops=dict(facecolor="lightseagreen", color="k"),
#                          capprops=dict(color="k"),
#                          whiskerprops=dict(color="k"),
#                          flierprops=dict(markerfacecolor="paleturquoise", markeredgecolor="k", marker='o'),
#                          medianprops=dict(color="k"))
# bp_fixed = ax[0].boxplot(x=logLts_fixed, showfliers=show_fliers, positions=fixed_positions, patch_artist=True,
#                          boxprops=dict(facecolor="indianred", color="k"),
#                          capprops=dict(color="k"),
#                          whiskerprops=dict(color="k"),
#                          flierprops=dict(markerfacecolor="lightcoral", markeredgecolor="k"),
#                          medianprops=dict(color="k"))
# ax[0].set_xticks(positions)
# ax[0].set_xticklabels(["{:.4f}".format(eps) for eps in initial_epsilons], rotation=0)
# for vert_line_pos in vertical_lines:
#     ax[0].axvline(x=vert_line_pos, linestyle='--', color='darkgrey')
# ax[0].set_ylabel("Log Normalizing Constant", fontsize=13)
# ax[0].legend(handles=[bp_adapt["boxes"][0], bp_fixed["boxes"][0]], labels=["Adaptive", "Non-Adaptive"], loc="lower center", fontsize=9)
# # Second plot
# bp_adapt_only = ax[1].boxplot(x=logLts_adapt, showfliers=show_fliers, positions=adapt_positions, patch_artist=True,
#                               boxprops=dict(facecolor="lightseagreen", color="k"),
#                               capprops=dict(color="k"),
#                               whiskerprops=dict(color="k"),
#                               flierprops=dict(markerfacecolor="paleturquoise", markeredgecolor="k", marker='o'),
#                               medianprops=dict(color="k"))
# ax[1].set_xticks(positions)
# ax[1].set_xticklabels(["{:.4f}".format(eps) for eps in initial_epsilons], rotation=0)
# for vert_line_pos in vertical_lines:
#     ax[1].axvline(x=vert_line_pos, linestyle='--', color='darkgrey')
# ax[1].set_xlabel(r"$\mathregular{\theta_0}$" + " (Initial Epsilons)", fontsize=13)
# ax[1].set_ylabel("Log Normalizing Constant", fontsize=13)
# ax[1].legend(handles=[bp_adapt_only["boxes"][0]], labels=["Adaptive"], loc='lower center', fontsize=9)
# plt.tight_layout()
# # plt.savefig("boxplots_log_normalizing_constant_new.png")
# plt.show()


eps_means = np.array([res['out']['epsilon_params_history'][-1]['mean'] for res in results_adapt]).reshape(n_runs, n_eps)
show_fliers = True
fig, ax = plt.subplots(figsize=(15, 4))
bp_adapt = ax.boxplot(x=eps_means, showfliers=show_fliers, positions=adapt_positions, patch_artist=True,
                      boxprops=dict(facecolor="lightseagreen", color="k"),
                      capprops=dict(color="k"),
                      whiskerprops=dict(color="k"),
                      flierprops=dict(markerfacecolor="paleturquoise", markeredgecolor="k"),
                      medianprops=dict(color="k"))
ax.set_xticks(positions)
ax.set_xticklabels(["{:.4f}".format(eps) for eps in initial_epsilons], rotation=0)
for vert_line_pos in vertical_lines:
    ax.axvline(x=vert_line_pos, linestyle='--', color='darkgrey')
ax.set_xlabel("Initial Epsilons", fontsize=13)
ax.set_ylabel("Final Epsilon Mean", fontsize=13)
ax.legend(handles=[bp_adapt["boxes"][0]], labels=["Adaptive"], loc='upper center', fontsize=9)
plt.savefig("boxplots_final_eps_mean_new.png")
plt.show()
