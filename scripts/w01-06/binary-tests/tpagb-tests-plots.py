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

ref_EAGB = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/compare-2-msun/vassiliadis-woods/LOGS/EAGB/history.data"
)

ref = mr.MesaData(
    f"/home/koen/master-internship/mesa-models/compare-2-msun/vassiliadis-woods/LOGS/TPAGB/history.data"
)

binary_histories = []
star_histories = []
for model_id in range(1, 10):
    binary_histories.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test/model-{model_id}/binary_history.data"
        )
    )

    star_histories.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test/model-{model_id}/LOGS/TPAGB/history.data"
        )
    )

for model_id in range(1, 7):
    match model_id:
        case 1:
            binary_histories.insert(
                6,
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-2/model-{model_id}/binary_history.data"
                ),
            )
            star_histories.insert(
                6,
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-2/model-{model_id}/LOGS/TPAGB/history.data"
                ),
            )

        case _:
            binary_histories.append(
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-2/model-{model_id}/binary_history.data"
                ),
            )
            star_histories.append(
                mr.MesaData(
                    f"/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test-2/model-{model_id}/LOGS/TPAGB/history.data"
                )
            )

# %%

lists = [
    [0, 1, 2],
    [3, 4, 5, 6],
    [11, 12, 13, 14],
    [7, 8, 9, 10],
]
for j, l in enumerate(lists):
    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
    )

    style = dict(
        c="C10",
        alpha=0.7,
        linewidth=4,
        zorder=0,
    )

    for i, id in enumerate(l):
        if j == 0:
            posneg = 1
        else:
            posneg = -1
        history = binary_histories[id]
        star_history = star_histories[id]

        index = np.argwhere(ref_EAGB.last_saved_R >= 113.62)[0][0]

        axs[0].plot(star_history.age, star_history.R, c=f"C{i}", zorder=10 + posneg * i)
        axs[0].plot(
            history.age, history.rl_1, zorder=0, c=f"C{i}", alpha=0.5, linewidth=2
        )

        lims = axs[0].get_xlim()
        limsy = axs[0].get_ylim()

        axs[1].plot(
            history.age, history.lg_mstar_dot_1, c=f"C{i}", zorder=10 + posneg * i
        )

        axs[2].plot(
            star_history.star_age,
            star_history.star_mass - star_history.he_core_mass,
            c=f"C{i}",
            label=f"$q={1-i*.25:.2f}$",
            zorder=10 + posneg * i,
        )

    axs[0].plot(
        ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
        ref_EAGB.R[index:],
        **style,
    )
    axs[0].plot(
        ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
        ref.R,
        **style,
    )

    axs[1].plot(
        ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
        ref_EAGB.log_abs_mdot[index:],
        **style,
    )
    axs[1].plot(
        ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
        ref.log_abs_mdot,
        **style,
    )

    axs[2].plot(
        ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
        ref_EAGB.star_mass[index:] - ref_EAGB.he_core_mass[index:],
        **style,
    )
    axs[2].plot(
        ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
        ref.star_mass - ref.he_core_mass,
        **style,
    )

    # axs[1].set_yscale("log")
    axs[2].legend()
    axs[2].set_xlim(lims[0], 1.1 * lims[1])
    axs[0].set_ylim(limsy[0], limsy[1])
    axs[0].set_ylabel(r"$R$ ($R_\odot$)")
    axs[1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
    axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
    axs[2].set_xlabel("TPAGB age (yr)")
    fig.text(
        0.17,
        0.95,
        f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}\\, R_\\odot$",
        ha="left",
        va="bottom",
        fontsize=8,
    )

    plt.savefig(
        f"/home/koen/LaTeX-setup/plots/tpagb-binary-test-{j+1}.pgf", format="pgf"
    )
    plt.show()
    plt.close()

# %%
fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)


for id in [0, 3, 11, 7]:
    history = binary_histories[id]
    star_history = star_histories[id]

    index = np.argwhere(ref_EAGB.last_saved_R >= 113.62)[0][0]

    axs[0].plot(
        star_history.age,
        star_history.R,
        zorder=10 - id,
        label=f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}\\, R_\\odot$",
    )

    axs[1].plot(history.age, history.lg_mstar_dot_1, zorder=10 - id)

    axs[2].plot(
        star_history.star_age,
        star_history.star_mass - star_history.he_core_mass,
        zorder=10 - id,
    )

style = dict(
    c="C8",
    alpha=0.7,
    linewidth=4,
    zorder=0,
)


axs[0].plot(
    ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
    ref_EAGB.R[index:],
    **style,
    label="Reference single star",
)
axs[0].plot(
    ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
    ref.R,
    **style,
)

axs[1].plot(
    ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
    ref_EAGB.log_abs_mdot[index:],
    **style,
)
axs[1].plot(
    ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
    ref.log_abs_mdot,
    **style,
)

axs[2].plot(
    ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
    ref_EAGB.star_mass[index:] - ref_EAGB.he_core_mass[index:],
    **style,
)
axs[2].plot(
    ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
    ref.star_mass - ref.he_core_mass,
    **style,
)


axs[0].legend(ncols=2, frameon=False)
# axs[1].set_yscale("log")
axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
axs[2].set_xlabel("TPAGB age (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/tpagb-binary-test-all.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)
for id in [0, 3, 11, 7]:
    history = binary_histories[id]
    star_history = star_histories[id]
    axs[0].plot(
        history.age,
        history.lg_mstar_dot_1,
        zorder=10 - id,
        label=f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}\\, R_\\odot$",
    )
    axs[1].plot(history.age, history.period_days)

print(history.bulk_names)
axs[0].legend(ncols=2, frameon=False)
fig.text(
    0.17,
    0.92,
    f"$q = 1$",
    ha="left",
    va="bottom",
    fontsize=8,
)
axs[0].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
axs[0].set_ylim(-12, 3)
axs[1].set_ylabel("Period (days)")
axs[1].set_xlabel("TPAGB age (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/period.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
for id in [8]:
    history = binary_histories[id]

    plt.plot(history.age, history.lg_wind_mdot_1, label="wind")
    plt.plot(
        history.age,
        history.lg_mtransfer_rate,
        linewidth=3,
        zorder=0,
        alpha=0.5,
        label="RLOF",
    )

print(history.bulk_names)
plt.xlabel("TPAGB age (yr)")
plt.ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
fig.text(
    0.37,
    0.86,
    f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}$, $q=0.50$",
    ha="left",
    va="bottom",
    fontsize=8,
)
plt.ylim(-32.5, 0)
plt.legend()
plt.savefig("/home/koen/LaTeX-setup/plots/wind-vs-transfer.pgf", format="pgf")
plt.show()
plt.close()
# %%

model = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test/model-2/LOGS/TPAGB/profile1.data"
)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
history = binary_histories[1]
star_history = star_histories[1]

print(star_history.bulk_names)
plt.plot(np.log10(model.Rho / ((model.T / 1e6) ** 3)), model.logT)

plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/fail-1.pgf", format="pgf")
plt.show()
plt.close()
# %%


model = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tests/binary-tpagb-test/model-2/LOGS/TPAGB/profile1.data"
)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)
delta = [26200, 26200, 241800, 500900]
for i, id in enumerate([1, 2, 6, 11]):
    history = binary_histories[id]
    star_history = star_histories[id]

    plt.plot(
        star_history.star_age - delta[i],
        star_history.R,
        c=f"C{i}",
        label=f"$q={(history.star_2_mass[0] / history.star_1_mass[0]):.2f}$",
    )
    plt.plot(history.age - delta[i], history.rl_1, c=f"C{i}", alpha=0.5, linewidth=2)

plt.legend(ncols=2, loc="upper left", frameon=False)
plt.xlim(-50, 800)
plt.xlabel("Shifted time (yr)")
plt.ylabel("$R$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/fail-2.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    3, 2, sharex="col", figsize=set_size(full, height=0.7), constrained_layout=True
)

axs = axs.T

style = dict(
    c="C8",
    alpha=0.7,
    linewidth=4,
    zorder=0,
)

j = 0
for k in [0, 1]:
    for i, id in enumerate([14]):
        if j == 0:
            posneg = 1
        else:
            posneg = -1
        history = binary_histories[id]
        star_history = star_histories[id]

        index = np.argwhere(ref_EAGB.last_saved_R >= 113.62)[0][0]

        axs[k][0].plot(
            star_history.age - 369790, star_history.R, c=f"C{i}", zorder=10 + posneg * i
        )
        axs[k][0].plot(
            history.age - 369790,
            history.rl_1,
            zorder=0,
            c=f"C{i}",
            alpha=0.5,
            linewidth=2,
        )

        lims = axs[k][0].get_xlim()
        limsy = axs[k][0].get_ylim()

        axs[k][1].plot(
            history.age - 369790,
            history.lg_mstar_dot_1,
            c=f"C{i}",
            zorder=10 + posneg * i,
        )

        axs[k][2].plot(
            star_history.star_age - 369790,
            star_history.star_mass - star_history.he_core_mass,
            c=f"C{i}",
            label=f"$q={1-i*.25:.2f}$",
            zorder=10 + posneg * i,
        )
    axs[k][2].set_xlim(lims[0], 1.1 * lims[1])
    axs[k][0].set_ylim(limsy[0], limsy[1])
    axs[k][1].set_ylim(*axs[k][1].get_ylim())
    axs[k][2].set_ylim(*axs[k][2].get_ylim())


for ax in axs[1]:
    ax.yaxis.tick_right()
# axs[1].set_yscale("log")
axs[0][0].set_ylabel(r"$R$ ($R_\odot$)")
axs[0][1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
axs[0][2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
axs[0][2].set_xlabel("Time (yr)")
axs[1][2].set_xlabel("Time (yr)")
fig.text(
    0.17,
    0.95,
    f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}\\, R_\\odot$",
    ha="left",
    va="bottom",
    fontsize=8,
)

axs[0][2].set_xlim(369000 - 369790, 369850 - 369790)
axs[1][2].set_xlim(369780 - 369790, 369805 - 369790)

for i in [0, 1]:
    for j, ax in enumerate(axs[i]):
        ax.fill_between(
            axs[1][2].get_xlim(),
            [axs[1][j].get_ylim()[0], axs[1][j].get_ylim()[0]],
            [axs[1][j].get_ylim()[1], axs[1][j].get_ylim()[1]],
            color="C8",
            alpha=0.3,
            zorder=-10,
        )

plt.savefig(f"/home/koen/LaTeX-setup/plots/tpagb-binary-test-zoom-1.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

delta = [762100, 762250, 886100, 886000][::-1]
for i, id in enumerate([7, 8, 9, 10]):
    history = binary_histories[id]
    star_history = star_histories[id]
    axs[0].plot(
        star_history.age - delta[i],
        star_history.R,
        zorder=10 - id,
        label=f"$q = {1 - 0.25*i:.2f}$",
    )
    axs[0].plot(
        history.age - delta[i],
        history.rl_1,
        c=f"C{i}",
        alpha=0.5,
        zorder=0,
        linewidth=2,
    )
    axs[1].plot(
        history.age - delta[i],
        history.lg_mstar_dot_1,
        zorder=10 - id,
        label=f"$q = {1 - 0.25*i:.2f}$",
    )
    axs[2].plot(history.age - delta[i], history.period_days)

print(history.bulk_names)
axs[0].legend(ncols=2, frameon=False)
fig.text(
    0.17,
    0.42,
    f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}\\, R_\\odot$",
    ha="left",
    va="bottom",
    fontsize=8,
)
axs[0].grid(axis="x")
axs[1].grid(axis="x")
axs[2].grid(axis="x")
axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
axs[2].set_xlim(-50, 200)
axs[2].set_ylabel("Period (days)")
axs[2].set_xlabel("Shifted time (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/period-q-zoom.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

for i, id in enumerate([7, 8, 9, 10]):
    history = binary_histories[id]
    star_history = star_histories[id]
    axs[0].plot(
        history.age,
        history.lg_mstar_dot_1,
        zorder=10 - id,
        label=f"$q = {1 - 0.25*i:.2f}$",
    )
    axs[1].plot(history.age, history.period_days)

print(history.bulk_names)
axs[0].legend(ncols=2, frameon=False)
fig.text(
    0.17,
    0.47,
    f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}\\, R_\\odot$",
    ha="left",
    va="bottom",
    fontsize=8,
)
axs[0].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
axs[1].set_ylabel("Period (days)")
axs[1].set_xlabel("Shifted time (yr)")
plt.savefig("/home/koen/LaTeX-setup/plots/period-q.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

style = dict(
    c="C8",
    alpha=0.7,
    linewidth=4,
    zorder=0,
)

for i, id in enumerate([4]):
    history = binary_histories[id]
    star_history = star_histories[id]

    index = np.argwhere(ref_EAGB.last_saved_R >= 113.62)[0][0]

    (l0,) = axs[0].plot(
        star_history.age,
        star_history.R,
        c=f"C{i}",
        zorder=10 + i,
        label="Donor star",
    )
    (l1,) = axs[0].plot(
        history.age,
        history.rl_1,
        zorder=0,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
        label="Roche lobe donor star",
    )

    lims = axs[0].get_xlim()
    limsy = axs[0].get_ylim()

    axs[1].plot(history.age, history.lg_mstar_dot_1, c=f"C{i}", zorder=10 + i)

    axs[2].plot(
        star_history.star_age,
        star_history.star_mass - star_history.he_core_mass,
        c=f"C{i}",
        label=f"$q={1-i*.25:.2f}$",
        zorder=10 + i,
    )

(l2,) = axs[0].plot(
    ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
    ref_EAGB.R[index:],
    **style,
    label="Reference single star",
)
axs[0].plot(
    ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
    ref.R,
    **style,
)

axs[1].plot(
    ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
    ref_EAGB.log_abs_mdot[index:],
    **style,
)
axs[1].plot(
    ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
    ref.log_abs_mdot,
    **style,
)

axs[2].plot(
    ref_EAGB.star_age[index:] - ref_EAGB.star_age[index],
    ref_EAGB.star_mass[index:] - ref_EAGB.he_core_mass[index:],
    **style,
)
axs[2].plot(
    ref.star_age + ref_EAGB.star_age[-1] - ref_EAGB.star_age[index],
    ref.star_mass - ref.he_core_mass,
    **style,
)

# axs[1].set_yscale("log")
axs[2].legend(loc="upper right", handles=[l0, l1, l2])
axs[2].set_xlim(lims[0], 1.1 * lims[1])
axs[0].set_ylim(limsy[0], limsy[1])
axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
axs[2].set_xlabel("TPAGB age (yr)")

print(history.period_days[0], history.period_days[-1])

plt.savefig(f"/home/koen/LaTeX-setup/plots/slide-deck.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

style = dict(
    c="C8",
    alpha=0.7,
    linewidth=4,
    zorder=0,
)

for i, id in enumerate(range(7, 11)):
    history = binary_histories[id]
    star_history = star_histories[id]

    index = np.argwhere(ref_EAGB.last_saved_R >= 113.62)[0][0]

    (l0,) = axs[0].plot(
        star_history.star_mass - star_history.he_core_mass,
        star_history.R,
        c=f"C{i}",
        zorder=10 + i,
        label="Donor star",
    )
    (l1,) = axs[0].plot(
        star_history.star_mass - star_history.he_core_mass,
        star_history.rl_1,
        zorder=0,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
        label="Roche lobe donor star",
    )

    lims = axs[0].get_xlim()
    limsy = axs[0].get_ylim()

    axs[1].plot(
        star_history.star_mass - star_history.he_core_mass,
        star_history.lg_mstar_dot_1,
        c=f"C{i}",
        zorder=10 + i,
    )

    axs[1].plot(
        star_history.star_mass - star_history.he_core_mass,
        star_history.lg_wind_mdot_1,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
    )

    axs[2].plot(
        star_history.star_mass - star_history.he_core_mass,
        star_history.period_days,
        c=f"C{i}",
        label=f"$q={1-i*.25:.2f}$",
        zorder=10 + i,
    )

axs[2].set_xlim(lims[1], lims[0])
axs[0].set_ylim(limsy[0], limsy[1])
axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
axs[2].set_xlabel("TPAGB age (yr)")

print(history.period_days[0], history.period_days[-1])

# plt.savefig(f"/home/koen/LaTeX-setup/plots/mass-test.pgf", format="pgf")
plt.show()
plt.close()
# %%


fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=0.7), constrained_layout=True
)

style = dict(
    c="C8",
    alpha=0.7,
    linewidth=4,
    zorder=0,
)

for i, id in enumerate(range(7, 11)):
    history = binary_histories[id]
    star_history = star_histories[id]

    index = np.argwhere(ref_EAGB.last_saved_R >= 113.62)[0][0]
    env_mass = star_history.star_mass - star_history.he_core_mass
    env_mass = np.log10(env_mass / env_mass[0])

    (l0,) = axs[0].plot(
        env_mass,
        star_history.R,
        c=f"C{i}",
        zorder=10 + i,
        label="Donor star",
    )
    (l1,) = axs[0].plot(
        env_mass,
        star_history.rl_1,
        zorder=0,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
        label="Roche lobe donor star",
    )

    lims = axs[0].get_xlim()
    limsy = axs[0].get_ylim()

    axs[1].plot(
        env_mass,
        star_history.lg_mstar_dot_1,
        c=f"C{i}",
        zorder=10 + i,
    )

    axs[1].plot(
        env_mass,
        star_history.lg_wind_mdot_1,
        c=f"C{i}",
        alpha=0.4,
        linewidth=3,
    )

    axs[2].plot(
        env_mass,
        star_history.period_days,
        c=f"C{i}",
        label=f"$q={1-i*.25:.2f}$",
        zorder=10 + i,
    )

axs[2].set_xlim(lims[1], lims[0])
axs[0].set_ylim(limsy[0], limsy[1])
axs[0].set_ylabel(r"$R$ ($R_\odot$)")
axs[1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
axs[2].set_ylabel(r"$M_\textrm{env}$ ($M_\odot$)")
axs[2].set_xlabel("TPAGB age (yr)")

print(history.period_days[0], history.period_days[-1])

# plt.savefig(f"/home/koen/LaTeX-setup/plots/mass-test.pgf", format="pgf")
plt.show()
plt.close()
# %%

lists = [
    [0, 1, 2],
    [3, 4, 5, 6],
    [11, 12, 13, 14],
    [7, 8, 9, 10],
]
for j, l in enumerate(lists):
    fig, axss = plt.subplots(
        3,
        2,
        sharex=True,
        figsize=set_size(full, height=0.8),
        constrained_layout=True,
        width_ratios=[1.6, 1],
    )

    axs = axss[:, 0][::-1]
    laxs = axss[:, 1][::-1]

    for lax in laxs:
        lax.axis("off")
    style = dict(
        c="C8",
        alpha=0.7,
        linewidth=4,
        zorder=0,
    )

    ls = []
    for i, id in enumerate(l):
        if j == 0:
            posneg = 1
        else:
            posneg = -1
        history = binary_histories[id]
        star_history = star_histories[id]

        index = np.argwhere(ref_EAGB.last_saved_R >= 113.62)[0][0]
        env_mass = star_history.star_mass - star_history.he_core_mass
        # env_mass = np.log10(env_mass / env_mass[0])

        (l,) = axs[0].plot(
            env_mass,
            star_history.R,
            c=f"C6",
            zorder=0,
            label="Star radius",
        )
        axs[0].plot(
            env_mass,
            star_history.R,
            c=f"C{i}",
            zorder=10 + posneg * i,
            label="Star radius",
        )
        (l2,) = axs[0].plot(
            env_mass[0],
            star_history.rl_1[0],
            zorder=0,
            c=f"C6",
            alpha=0.5,
            linewidth=3,
            label="Roche lobe radius",
        )
        axs[0].plot(
            env_mass,
            star_history.rl_1,
            zorder=0,
            c=f"C{i}",
            alpha=0.5,
            linewidth=3,
            label="Roche lobe radius",
        )
        laxs[0].legend(
            handles=[l, l2],
            loc="center left",
        )

        lims = axs[0].get_xlim()
        limsy = axs[0].get_ylim()

        (l,) = axs[1].plot(
            env_mass[0],
            star_history.lg_mstar_dot_1[0],
            c=f"C6",
            zorder=10 + posneg * i,
            label="Total $\dot{M}$",
        )
        axs[1].plot(
            env_mass, star_history.lg_mstar_dot_1, c=f"C{i}", zorder=10 + posneg * i
        )
        (l2,) = axs[1].plot(
            env_mass[0],
            star_history.lg_wind_mdot_1[0],
            c=f"C6",
            alpha=0.4,
            linewidth=3,
            label="Wind $\dot{M}$",
        )
        axs[1].plot(
            env_mass,
            star_history.lg_wind_mdot_1,
            c=f"C{i}",
            alpha=0.4,
            linewidth=3,
        )

        laxs[1].legend(
            handles=[l, l2],
            loc="center left",
        )

        (l,) = axs[2].plot(
            env_mass,
            star_history.period_days,
            c=f"C{i}",
            label=f"$q={1-i*.25:.2f}$",
            zorder=10 + i,
        )
        ls.append(l)
        laxs[2].legend(
            handles=ls,
            loc="center left",
        )
    # axs[1].set_yscale("log")
    axs[2].set_xlim(lims[1], lims[0])
    axs[0].set_ylim(limsy[0], limsy[1])
    axs[0].set_ylabel(r"$R$ ($R_\odot$)")
    axs[1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
    axs[2].set_ylabel(r"Orbital period (days)")
    axs[0].set_xlabel(r"Envelope mass ($M_\odot$)")
    fig.text(
        0.11,
        0.95,
        f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}\\, R_\\odot$",
        ha="left",
        va="bottom",
        fontsize=8,
    )

    plt.savefig(
        f"/home/koen/LaTeX-setup/plots/tpagb-binary-mass-{j+1}.pgf", format="pgf"
    )
    plt.show()
    plt.close()
# %%

lists = [
    [0, 1, 2],
    [3, 4, 5, 6],
    [11, 12, 13, 14],
    [7, 8, 9, 10],
]
for j, l in enumerate(lists):
    fig, axss = plt.subplots(
        3,
        2,
        sharex=True,
        figsize=set_size(full, height=0.8),
        constrained_layout=True,
        width_ratios=[1.6, 1],
    )

    axs = axss[:, 0][::-1]
    laxs = axss[:, 1][::-1]

    for lax in laxs:
        lax.axis("off")
    style = dict(
        c="C8",
        alpha=0.7,
        linewidth=4,
        zorder=0,
    )

    ls = []
    for i, id in enumerate(l):
        if j == 0:
            posneg = 1
        else:
            posneg = -1
        history = binary_histories[id]
        star_history = star_histories[id]

        index = np.argwhere(ref_EAGB.last_saved_R >= 113.62)[0][0]
        env_mass = star_history.star_mass - star_history.he_core_mass
        env_mass = np.log10(env_mass / env_mass[0])

        (l,) = axs[0].plot(
            env_mass,
            star_history.R,
            c=f"C6",
            zorder=0,
            label="Star radius",
        )
        axs[0].plot(
            env_mass,
            star_history.R,
            c=f"C{i}",
            zorder=10 + posneg * i,
            label="Star radius",
        )
        (l2,) = axs[0].plot(
            env_mass[0],
            star_history.rl_1[0],
            zorder=0,
            c=f"C6",
            alpha=0.5,
            linewidth=3,
            label="Roche lobe radius",
        )
        axs[0].plot(
            env_mass,
            star_history.rl_1,
            zorder=0,
            c=f"C{i}",
            alpha=0.5,
            linewidth=3,
            label="Roche lobe radius",
        )
        laxs[0].legend(
            handles=[l, l2],
            loc="center left",
        )

        lims = axs[0].get_xlim()
        limsy = axs[0].get_ylim()

        (l,) = axs[1].plot(
            env_mass[0],
            star_history.lg_mstar_dot_1[0],
            c=f"C6",
            zorder=10 + posneg * i,
            label="Total $\dot{M}$",
        )
        axs[1].plot(
            env_mass, star_history.lg_mstar_dot_1, c=f"C{i}", zorder=10 + posneg * i
        )
        (l2,) = axs[1].plot(
            env_mass[0],
            star_history.lg_wind_mdot_1[0],
            c=f"C6",
            alpha=0.4,
            linewidth=3,
            label="Wind $\dot{M}$",
        )
        axs[1].plot(
            env_mass,
            star_history.lg_wind_mdot_1,
            c=f"C{i}",
            alpha=0.4,
            linewidth=3,
        )

        laxs[1].legend(
            handles=[l, l2],
            loc="center left",
        )

        (l,) = axs[2].plot(
            env_mass,
            star_history.period_days,
            c=f"C{i}",
            label=f"$q={1-i*.25:.2f}$",
            zorder=10 + i,
        )
        ls.append(l)
        laxs[2].legend(
            handles=ls,
            loc="center left",
        )
    # axs[1].set_yscale("log")
    axs[2].set_xlim(lims[1], lims[0])
    axs[0].set_ylim(limsy[0], limsy[1])
    axs[0].set_ylabel(r"$R$ ($R_\odot$)")
    axs[1].set_ylabel(r"$\log(\dot{M} / M_\odot \textrm{ yr}^{-1})$")
    axs[2].set_ylabel(r"Orbital period (days)")
    axs[0].set_xlabel(r"$\log(M_\textrm{env} / M_\textrm{env, ini})$")
    fig.text(
        0.11,
        0.95,
        f"$R_\\textrm{{RL}} = {history.rl_1[0]:.0f}\\, R_\\odot$",
        ha="left",
        va="bottom",
        fontsize=8,
    )

    plt.savefig(
        f"/home/koen/LaTeX-setup/plots/tpagb-binary-mass-{j+1}-log.pgf", format="pgf"
    )
    plt.show()
    plt.close()
# %%
