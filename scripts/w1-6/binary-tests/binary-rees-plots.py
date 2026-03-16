import mesa_reader as mr

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
# %%

binary_history = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-59R-q1/binary_history.data"
)

star_history = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-59R-q1/LOGS/EAGB/history.data"
)

# %%

models = []
for i in range(1, 36):
    try:
        models.append(
            mr.MesaData(
                f"/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-59R-q1/LOGS/EAGB/profile{i}.data"
            )
        )
    except:
        break
# %%
import matplotlib as mpl
import matplotlib.cm as cm

fig, axs = plt.subplots(
    3,
    2,
    sharex="col",
    figsize=set_size(full, height=1),
    sharey="row",
)

axs = axs.flatten()
model_numbers = []

norm = mpl.colors.Normalize(vmin=-5, vmax=-2.4)
cmap = cm.viridis  # or plasma, magma, etc.

norm2 = mpl.colors.Normalize(vmin=1.85, vmax=3)
cmap2 = cm.viridis  # or plasma, magma, etc.

for model in models[2:]:

    if model.model_number % 500 == 0 or model.model_number == 38639:
        continue
    model_numbers.append(model.model_number)
    superadiabatic = np.argwhere(model.grada_sub_gradT < -0.1)

    index = star_history.index_of_model_number(model.model_number)

    print(model.star_mass)
    color = cmap(norm(star_history.lg_mstar_dot_1[index]))
    color2 = cmap2(norm2(model.star_mass))
    axs[0].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        model.entropy[superadiabatic],
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[0].plot(np.log10(1 - model.mass / model.mass[0]), model.entropy, color=color)

    axs[1].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        model.entropy[superadiabatic],
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[1].plot(np.log10(1 - model.mass / model.mass[0]), model.entropy, color=color2)

    axs[2].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        model.thermal_time_to_surface[superadiabatic] / 31556926,
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[2].plot(
        np.log10(1 - model.mass / model.mass[0]),
        model.thermal_time_to_surface / 31556926,
        color=color,
    )

    axs[3].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        model.thermal_time_to_surface[superadiabatic] / 31556926,
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[3].plot(
        np.log10(1 - model.mass / model.mass[0]),
        model.thermal_time_to_surface / 31556926,
        color=color2,
    )

    axs[4].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        (model.star_mass - model.mass[superadiabatic])
        / model.thermal_time_to_surface[superadiabatic]
        * 31556926,
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[4].plot(
        np.log10(1 - model.mass / model.mass[0]),
        (model.star_mass - model.mass) / model.thermal_time_to_surface * 31556926,
        color=color,
    )

    axs[5].plot(
        np.log10(1 - model.mass[superadiabatic] / model.mass[0]),
        (model.star_mass - model.mass[superadiabatic])
        / model.thermal_time_to_surface[superadiabatic]
        * 31556926,
        linewidth=10,
        color="gray",
        zorder=0,
    )
    axs[5].plot(
        np.log10(1 - model.mass / model.mass[0]),
        (model.star_mass - model.mass) / model.thermal_time_to_surface * 31556926,
        color=color2,
    )

axs[0].set_ylim(15, 35)
axs[0].set_ylabel(r"$s / (N_\textrm{a} k_\textrm{B})$")

axs[1].set_ylim(15, 35)

axs[2].invert_xaxis()
axs[2].set_xlim(-8, 0)
axs[2].set_yscale("log")
axs[2].set_ylim(10**-6, 10**2)
axs[2].set_ylabel(r"$\tau_\textrm{th}$ (yr)")

axs[3].invert_xaxis()
axs[3].set_xlim(-8, 0)
axs[3].set_yscale("log")
axs[3].set_ylim(10**-6, 10**2)


axs[4].invert_xaxis()
axs[4].set_yscale("log")
axs[4].set_ylim(10**-5, 10**0)
axs[4].set_xlabel(r"$\log(1- m / M)$")
axs[4].set_ylabel(r"$\dot{M}_\textrm{th}$ ($M_\odot$ yr$^{-1}$)")

axs[5].invert_xaxis()
axs[5].set_yscale("log")
axs[5].set_ylim(10**-5, 10**0)
axs[5].set_xlabel(r"$\log(1- m / M)$")


sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(sm, ax=axs[0], pad=0.02, location="top")
cbar.set_label(r"$\log(\dot{M}_\textrm{d} / (M_\odot\textrm{ yr}^{-1}))$)")

sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)

cbar = fig.colorbar(sm, ax=axs[1], pad=0.02, location="top")
cbar.set_label(r"$M$ ($M_\odot$)")

plt.tight_layout()
plt.savefig("/home/koen/LaTeX-setup/plots/binary-rees-3MSUN-profiles.pgf", format="pgf")
plt.show()
plt.close()
# %%

index = []
for model in models[2:]:

    index.append(binary_history.index_of_model_number(model.model_number))


fig, axs = plt.subplots(
    1,
    2,
    figsize=set_size(column),
    constrained_layout=True,
)

axs[0].plot(binary_history.age / 1e3, binary_history.lg_mstar_dot_1)
axs[0].scatter(
    binary_history.age[index] / 1e3,
    binary_history.lg_mstar_dot_1[index],
    c=binary_history.star_1_mass[index],
    cmap="viridis",
    vmin=1.85,
    vmax=3,
    zorder=1000,
    s=15,
)
axs[0].set_ylim(-8, 0)
axs[0].set_xlim(220, 279)
axs[1].plot(binary_history.age / 1e3, binary_history.lg_mstar_dot_1)
axs[1].set_xlim(279, 1050)
axs[1].set_ylim(-8, 0)

axs[0].spines["right"].set_visible(False)
axs[1].spines["left"].set_visible(False)
# axs[0].yaxis.tick_left()
axs[1].yaxis.set_ticklabels([])
axs[1].yaxis.set_ticks([])

d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=axs[0].transAxes, color="C8", clip_on=False, linewidth=0.75)
axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
axs[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=axs[1].transAxes)  # switch to the bottom axes
axs[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
axs[1].plot((-d, +d), (-d, +d), **kwargs)

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

axs[0].set_ylabel(r"$\log(\dot{M} / (M_\odot \textrm{yr}^{-1}))$")
fig.text(0.495, 0.0, "Time (Kyr)", ha="center", va="bottom")
fig.text(
    0.175,
    0.85,
    r"$R_\textrm{initial}= 59.2 R_\odot$, $M_\textrm{a} / M_\textrm{d} = 1.0$",
    ha="left",
    va="top",
    fontsize=8,
)
sm = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)

cbar = fig.colorbar(sm, ax=axs[1], pad=0.02, location="right")
cbar.set_label(r"$M$ ($M_\odot$)")

fig.get_layout_engine().set(h_pad=0.2)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/mass-loss-binary-rees-3MSUN.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

binary_history2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-59R-q0.8/binary_history.data"
)

star_history2 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-59R-q0.8/LOGS/EAGB/history.data"
)
# %%

fig, axs = plt.subplots(
    1,
    2,
    figsize=set_size(column),
    constrained_layout=True,
)

index_shift = np.argwhere(binary_history.lg_mstar_dot_1 > -8)[0][0]
t_shift = binary_history.age[index_shift] / 1e3

index_shift2 = np.argwhere(binary_history2.lg_mstar_dot_1 > -8)[0][0]
t_shift2 = binary_history2.age[index_shift2] / 1e3

axs[0].plot(binary_history.age / 1e3 - t_shift, binary_history.lg_mstar_dot_1)
axs[0].plot(binary_history2.age / 1e3 - t_shift2, binary_history2.lg_mstar_dot_1)
axs[0].set_ylim(-8, 0)
axs[0].set_xlim(0, 79)
axs[1].plot(
    binary_history.age / 1e3 - t_shift,
    binary_history.lg_mstar_dot_1,
    label=r"$M_\textrm{a} / M_\textrm{d} = 1.0$",
)
axs[1].plot(
    binary_history2.age / 1e3 - t_shift2,
    binary_history2.lg_mstar_dot_1,
    label=r"$M_\textrm{a} / M_\textrm{d} = 0.8$",
)
axs[1].set_xlim(79, 850)
axs[1].set_ylim(-8, 0.5)
axs[1].legend()

axs[0].spines["right"].set_visible(False)
axs[1].spines["left"].set_visible(False)
# axs[0].yaxis.tick_left()
axs[1].yaxis.set_ticklabels([])
axs[1].yaxis.set_ticks([])

d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=axs[0].transAxes, color="C8", clip_on=False, linewidth=0.75)
axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
axs[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=axs[1].transAxes)  # switch to the bottom axes
axs[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
axs[1].plot((-d, +d), (-d, +d), **kwargs)

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

axs[0].set_ylabel(r"$\log(\dot{M} / (M_\odot \textrm{yr}^{-1}))$")
fig.text(0.520, 0.0, "Time (Kyr)", ha="center", va="bottom")
fig.text(
    0.175,
    0.85,
    r"$R_\textrm{initial}= 59.2 R_\odot$",
    ha="left",
    va="top",
    fontsize=8,
)
fig.get_layout_engine().set(h_pad=0.2)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/mass-loss-binary-rees-3MSUN-compare-q.pgf",
    format="pgf",
)
plt.show()
plt.close()
# %%


binary_history3 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-115R-q1/binary_history.data"
)

star_history3 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-115R-q1/LOGS/EAGB/history.data"
)

binary_history4 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-115R-q0.8/binary_history.data"
)

star_history4 = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-rees-3M-115R-q0.8/LOGS/EAGB/history.data"
)

# %%

fig, axs = plt.subplots(
    1,
    1,
    figsize=set_size(column),
    constrained_layout=True,
)

index_shift3 = np.argwhere(binary_history3.lg_mstar_dot_1 > -8)[0][0]
t_shift3 = binary_history3.age[index_shift3] / 1e3

index_shift4 = np.argwhere(binary_history4.lg_mstar_dot_1 > -8)[0][0]
t_shift4 = binary_history4.age[index_shift4] / 1e3

axs.plot(
    binary_history3.age / 1e3 - t_shift3,
    binary_history3.lg_mstar_dot_1,
    label=r"$M_\textrm{a} / M_\textrm{d} = 1.0$",
)
axs.plot(
    binary_history4.age / 1e3 - t_shift4,
    binary_history4.lg_mstar_dot_1,
    label=r"$M_\textrm{a} / M_\textrm{d} = 0.8$",
)
axs.set_ylim(-9, 0.5)
axs.set_xlim(0, 185)
axs.set_ylabel(r"$\log(\dot{M} / (M_\odot \textrm{yr}^{-1}))$")
axs.set_xlabel(r"Time (Kyr)")
axs.legend()
plt.text(
    0.075,
    0.91,
    r"$R_\textrm{initial}= 116.0 R_\odot$",
    ha="left",
    va="top",
    transform=axs.transAxes,
    fontsize=8,
)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/mass-loss-binary-rees-3MSUN-compare-q-2.pgf",
    format="pgf",
)
plt.show()
plt.close()
# %%
