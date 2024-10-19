import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import numpy as np
import pickle
import matplotlib.patches as mpatches
rc('font', **{'family': 'STIXGeneral'})

with open("results/new_epsdist_cox400_seed6213879936_N500_T30_runs20_from1e-06_to100dot0_skewness3_aooFalse_skipoFalse_adaptepsTrue_adaptTFalse_massschedule.pkl", "rb") as file:
    results = pickle.load(file)

initial_epsilons = [0.0000001, 0.001, 10.0, 100.0]
n_eps = len(initial_epsilons)

eps_histories = []
for res in results:
    eps_histories.append(res['out']['epsilons'])

all_gammas = []
for res in results:
    all_gammas.append(res['out']['gammas'])


eps_to_show = [1e-6, 0.001, 10.0, 100.0]
n_eps_to_show = len(eps_to_show)
showfliers = True
y_log_scale = True
fig, ax = plt.subplots(nrows=n_eps_to_show, figsize=(8, 1.5*n_eps_to_show), sharex=True, sharey=False)
for _eps_ix, eps in enumerate(eps_to_show):
    eps_ix = _eps_ix  # initial_epsilons.index(eps)
    for p_ix, eps_hist in enumerate(eps_histories[eps_ix]):
        ax[_eps_ix].boxplot(x=eps_hist, showfliers=showfliers, positions=[p_ix], patch_artist=True,
                            boxprops=dict(facecolor="lightseagreen", color="k"),
                            capprops=dict(color="k"),
                            whiskerprops=dict(color="k"),
                            flierprops=dict(markerfacecolor="paleturquoise", markeredgecolor="k", markersize=2.5),
                            medianprops=dict(color="k"),
                            widths=0.8)
    if y_log_scale:
        ax[_eps_ix].set_yscale('log')
        ax[_eps_ix].set_yticks([0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    ax[_eps_ix].set_ylabel(r"$\mathregular{\epsilon}$", fontsize=15)
    ax[_eps_ix].grid(True, color='gainsboro')
    ax[_eps_ix].tick_params(axis='both', which='major', labelsize=8)
ax[-1].set_xlabel("Iteration", fontsize=15)
plt.tight_layout(h_pad=0.05)
# plt.savefig("distribution_epsilons_over_iterations_1em6_1em3_10_1000_new.png")
plt.show()
