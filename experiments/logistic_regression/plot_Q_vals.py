import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import numpy as np
import pickle
rc('font', **{'family': 'STIXGeneral'})

with open("results/Qvals_seed3729307047_N2500_T30_runs1.pkl", "rb") as file:
    results = pickle.load(file)

Q_vals_history = results[0]['out']['Q_vals_history']
gammas = results[0]['out']['gammas']
epsilon_values = np.geomspace(start=0.001, stop=10.0, num=18)
gamma_indices = [0, 28, 35, -1]


# Plot Q as a function of epsilon for 4 different gammas
# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(6, 6), sharex=True)
# for idx, gamma_idx in enumerate(gamma_indices):
#     ax[idx // 2, idx % 2].plot(epsilon_values, Q_vals_history[gamma_idx], marker='o', color='lightcoral', markeredgecolor='firebrick', lw=2)
#     ax[idx // 2, idx % 2].set_xscale('log')
#     max_idx = np.argmax(Q_vals_history[gamma_idx])
#     ax[idx // 2, idx % 2].axvline(epsilon_values[max_idx], lw=2, ls='--', color='darkgrey', zorder=0)
#     # add gamma text
#     ax[idx // 2, idx % 2].text(
#         x=0.1, y=0.9,
#         s=r"$\mathregular{\gamma = " + f"{gammas[:-1][gamma_idx]: .2f}" + "}$",
#         transform=ax[idx // 2, idx % 2].transAxes, fontsize=12,
#         bbox=dict(facecolor='white', alpha=0.7)
#     )
#     if idx // 2 == 1:
#         ax[idx // 2, idx % 2].set_xlabel(r"$\mathregular{\epsilon}$", fontsize=15)
#     if idx % 2 == 0:
#         ax[idx // 2, idx % 2].set_ylabel(r"$\mathregular{\overline{\upsilon}_{\gamma_{n-1}, T}}$", fontsize=15)
#     ax[idx // 2, idx % 2].grid(True, color='gainsboro')
# plt.tight_layout()
# plt.savefig("Q_vals_four_gammas.png")
# plt.show()


# Plot Q(epsilon) as a function of gamma
cmap = plt.get_cmap("viridis", len(epsilon_values))
norm = mpl.colors.LogNorm(vmin=epsilon_values.min(), vmax=epsilon_values.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for the colorbar
fig, ax = plt.subplots()
for eps_idx, eps in enumerate(epsilon_values):
    ax.plot(gammas[:-1], np.vstack(Q_vals_history)[:, eps_idx], color=cmap(norm(eps)))
ax.set_yscale('log')
ax.set_ylim(bottom=5*10**(-4), top=70000)
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label(r"$\mathregular{\epsilon}$", fontsize=15)
ax.grid(True, color='gainsboro')
ax.set_xlabel(r"$\mathregular{\gamma}$", fontsize=15)
ax.set_ylabel(r"$\mathregular{\overline{\upsilon}_{\gamma, T}}$", fontsize=15)
plt.savefig("Q_vals_function_of_gamma.png")
plt.show()


