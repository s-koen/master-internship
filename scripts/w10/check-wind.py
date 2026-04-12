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

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid

Rsun = 6.957e10
Msun = 1.3271244e26 / 6.67430e-8
G = 6.67430e-8


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 10))  # outward by 10 points
        else:
            spine.set_color("none")  # don't draw spine

    # turn off ticks where there is no spine
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


# %%
wind_sal = mr.MesaData(f"{MASTER}/wind/4/LOGS/TPAGB/history.data")
# %%
wind_sal2 = mr.MesaData(f"{MASTER}/wind/5/LOGS/TPAGB/history.data")
# %%
wind_sal3 = mr.MesaData(f"{MASTER}/wind/6/LOGS/TPAGB/history2.data")
# %%
wind_sal4 = mr.MesaData(f"{MASTER}/wind/6/LOGS/TPAGB/history.data")
# %%
wind_sal5 = mr.MesaData(f"{MASTER}/wind/7/LOGS/TPAGB/history.data")
# %%
wind_sal6 = mr.MesaData(f"{MASTER}/wind/8/LOGS/TPAGB/history.data")
# %%
wind_sal7 = mr.MesaData(f"{MASTER}/wind/9/LOGS/TPAGB/history.data")
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

axs.plot(hist.age, hist.Omega_star, c="C0", label=r"$\Omega_\textrm{spin}$")
axs.set_xlabel("Time (yr)")
axs.set_ylabel(r"Angular frequency (rad/sec)", color="C0")
axs.tick_params(axis="y", labelcolor="C0")

# Create a second y-axis for electricity consumption
ax = axs.twinx()
ax.plot(hist.age, hist.R, c="C2", label="Star radius")
ax.set_ylabel("Radius ($R_\odot$)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")

fig.legend(loc="outside upper center", ncols=4)
# plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-5.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

axs.plot(hist.age, hist.Omega_star, c="C0", label=r"$\Omega_\textrm{spin}$")
axs.plot(
    hist.age, hist.Omega_orb, c="C0", label=r"$\Omega_\textrm{orbit}$", linestyle="--"
)
axs.set_xlabel("Time (yr)")
axs.set_ylabel(r"Angular frequency (rad/sec)", color="C0")
axs.tick_params(axis="y", labelcolor="C0")

# Create a second y-axis for electricity consumption
ax = axs.twinx()
ax.plot(hist.age, hist.rl_1, c="C2", label="Roche Lobe radius")
ax.set_ylabel("Radius ($R_\odot$)", color="C2")
ax.tick_params(axis="y", labelcolor="C2")

fig.legend(loc="outside upper center", ncols=4)
plt.savefig("/home/koen/LaTeX-setup/plots/w8-tides-6.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

# plt.plot(hist1.age, hist1.R, c="C9")
plt.plot(
    default_wind.age,
    default_wind.rl_1,
    label="MESA default",
)
plt.plot(
    wind.age,
    wind.rl_1,
    label="Fast isotropic wind momentum loss",
    alpha=0.4,
    linewidth=6,
    zorder=-1,
    c="C2",
)
plt.plot(
    no_wind.age,
    no_wind.rl_1,
    label="No wind angular momentum loss",
    linestyle=":",
    linewidth=3,
    c="C3",
)
plt.plot(no_wind.age, no_wind.R, c="C9", label="Star Radius")

fig.legend(loc="outside upper center", ncols=2)
plt.xlabel(r"Time (yr)")
plt.ylabel(r"Radius ($R_\odot$)")
plt.ylim(450, 750)
plt.savefig("/home/koen/LaTeX-setup/plots/w9-wind-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

# plt.plot(no_wind.age, no_wind.jdot_ml)
plt.plot(wind.age, -wind.jdot_ml)
plt.plot(wind_sal.age, -wind_sal.jdot_ml)

plt.yscale("log")
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

# plt.plot(hist1.age, hist1.R, c="C9")
plt.plot(
    wind.age,
    wind.rl_1,
    label="Fast isotropic wind momentum loss",
    zorder=-1,
    c="C2",
)
plt.plot(
    no_wind.age,
    no_wind.rl_1,
    label="No wind angular momentum loss",
    c="C3",
)
plt.plot(
    wind_sal.age,
    wind_sal.rl_1,
    label="Saladino (no accretion)",
    linestyle="-",
    c="C0",
)
plt.plot(no_wind.age, no_wind.R, c="C2", alpha=0.4, linewidth=6)
plt.plot(wind_sal.age, wind_sal.R, c="C0", alpha=0.5, linewidth=6)

fig.legend(loc="outside upper center", ncols=3)
plt.xlabel(r"Time (yr)")
plt.ylabel(r"Radius ($R_\odot$)")
plt.ylim(350, 750)
plt.savefig("/home/koen/LaTeX-setup/plots/w9-wind-2.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.plot(wind_sal.age, wind_sal.Omega_star)
plt.plot(wind_sal2.age, wind_sal2.Omega_star)
plt.plot(wind_sal3.age, wind_sal3.Omega_star)
plt.show()
# %%
plt.plot(wind_sal5.age, wind_sal5.rl_1)
plt.plot(wind_sal7.age, wind_sal7.rl_1)
plt.show()

# %%
plt.plot(wind_sal.age, wind_sal.jdot_ls)
plt.plot(wind_sal2.age, wind_sal2.jdot_ls)
plt.plot(wind_sal3.age, wind_sal3.jdot_ls)
plt.show()
# %%
plt.plot(wind_sal.age, wind_sal.Omega_star)
plt.plot(wind_sal.age, wind_sal.Omega_orb)
plt.plot(wind_sal2.age, wind_sal2.Omega_star)
plt.plot(wind_sal2.age, wind_sal2.Omega_orb)
plt.plot(wind_sal3.age, wind_sal3.Omega_star)
plt.plot(wind_sal3.age, wind_sal3.Omega_orb)

# plt.plot(wind_sal2.age, wind_sal2.Omega_star)
plt.show()
# %%

plt.plot(wind_sal3.age, wind_sal3.jdot_ml)
plt.plot(wind_sal.age, wind_sal.jdot_ml)
plt.show()
# %%
print(wind_sal.bulk_names)
# %%

plt.plot(wind_sal3.age, wind_sal3.lg_mtransfer_rate, alpha=0.5, linewidth=6, c="C9")
plt.plot(wind_sal3.age, wind_sal3.lg_mstar_dot_2)
plt.show()
# %%
plt.plot(wind_sal3.age, wind_sal3.R / wind_sal3.rl_1)
plt.show()

# %%
plt.plot(wind_sal3.age, wind_sal3.lg_mtransfer_rate)
plt.show()
# %%
plt.plot(wind_sal5.age, wind_sal5.rl_1)
plt.plot(wind_sal5.age, wind_sal5.R)
plt.plot(wind_sal7.age, wind_sal7.rl_1)
plt.plot(wind_sal7.age, wind_sal7.R)
plt.show()
# %%

plt.plot(wind_sal7.star_age, wind_sal7.star_2_mass)
plt.plot(wind_sal5.star_age, wind_sal5.star_2_mass)
plt.show()
# %%
wind_sal5.bulk_names
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

limits = [14000, len(wind_sal6.age)]
for i, model in enumerate([wind_sal5, wind_sal6]):

    index = np.argwhere(model.R / model.rl_1 > 1)[0][0]
    print(index)
    plt.xlabel("Time (yr)")
    limit = limits[i]

    (l1,) = plt.plot(
        model.star_age[:limit],
        model.beta_accretion[:limit],
        c="C0",
        label=r"$\beta_\textrm{Saladino}$",
    )
    if index < limit:
        plt.scatter(
            model.star_age[index],
            model.beta_accretion[index],
            c="C0",
        )
        plt.scatter(
            model.star_age[index],
            model.eta_accretion[index],
            c="C1",
        )
        plt.scatter(
            model.star_age[index],
            model.vwind_over_vorbit[index],
            c="C2",
        )
        plt.scatter(
            model.star_age[index],
            model.star_2_mass[index] / model.star_1_mass[index],
            c="C3",
        )

    (l3,) = plt.plot(
        model.star_age[:limit],
        model.eta_accretion[:limit],
        c="C1",
        label=r"$\eta_\textrm{Saladino}$",
    )

    (l5,) = plt.plot(
        model.star_age[:limit],
        model.vwind_over_vorbit[:limit],
        c="C2",
        label=r"$v_\textrm{wind} / v_\textrm{orbit}$",
    )

    (l6,) = plt.plot(
        model.star_age[:limit],
        model.star_2_mass[:limit] / model.star_1_mass[:limit],
        c="C3",
        label=r"$q = M_\textrm{a} / M_\textrm{d}$",
    )


axs.set_xlim(axs.get_xlim())
l2 = plt.hlines(
    0.3,
    -1e10,
    1e10,
    color="C0",
    linewidth=0.75,
    alpha=0.3,
    label=r"$\beta_\textrm{max}$",
)
l4 = plt.hlines(
    0.6,
    -1e10,
    1e10,
    color="C1",
    linewidth=0.75,
    alpha=0.3,
    label=r"$\eta_\textrm{max}$",
)


plt.ylim(0, 1.2)
fig.legend(loc="outside upper center", ncols=3, handles=[l1, l2, l3, l4, l5, l6])
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-1.pgf", format="pgf")
plt.show()
plt.close()

# %%

plt.plot(wind_sal5.star_1_mass, wind_sal5.star_2_mass)
plt.plot(wind_sal6.star_1_mass, wind_sal6.star_2_mass)
plt.show()
# %%

plt.plot(wind_sal5.star_age, wind_sal5.rl_1)
plt.plot(wind_sal6.star_age, wind_sal6.rl_1)
plt.show()

# %%

plt.plot(wind_sal5.star_age, wind_sal5.wind_mass_transfer)
plt.plot(wind_sal6.star_age, wind_sal6.wind_mass_transfer)
plt.show()

# %%
profiles = []
profile_indices = []
for i in [17, 2, 4, 11]:
    profile = mr.MesaData(f"{MASTER}/wind/7/LOGS/TPAGB/profile{i}.data")
    profile_indices.append(wind_sal5.index_of_model_number(profile.model_number))
    profiles.append(profile)


fig, axs = plt.subplots(
    2, 1, figsize=set_size(column, height=0.75), constrained_layout=True, sharex=True
)

plt.xlabel("Time (yr)")

index = np.argwhere(wind_sal5.R / wind_sal5.rl_1 > 1)[0][0]
axs[0].set_ylabel("$R_\\textrm{conv}$ ($R_\odot$)")
axs[1].set_ylabel("$R_\\textrm{star} -R_\\textrm{conv}$ ($R_\odot$)")

axs[0].plot(
    wind_sal5.star_age,
    wind_sal5.R_conv,
    c="C0",
)
axs[0].scatter(
    wind_sal5.star_age[index], wind_sal5.R_conv[index], color="C2", zorder=1000
)

for i, profile_index in enumerate(profile_indices):
    axs[0].scatter(
        wind_sal5.star_age[profile_index],
        wind_sal5.R_conv[profile_index],
        color=f"C{i}",
        zorder=1000,
        facecolors="none",
        s=20,
    )
    axs[1].scatter(
        wind_sal5.star_age[profile_index],
        wind_sal5.R[profile_index] - wind_sal5.R_conv[profile_index],
        color=f"C{i}",
        zorder=1000,
        facecolors="none",
        s=20,
    )
axs[0].set_xlim(axs[0].get_xlim())

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.R - wind_sal5.R_conv,
    c="C0",
)
axs[1].scatter(
    wind_sal5.star_age[index],
    wind_sal5.R[index] - wind_sal5.R_conv[index],
    c="C2",
    label=r"Start of RLOF",
    zorder=1000,
)

plt.legend()
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-2.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, profile in enumerate(profiles):
    plt.plot(10**profile.logR, profile.gradr, alpha=0.3, linewidth=3, c=f"C{i}")
    plt.plot(10**profile.logR, profile.gradT, c=f"C{i}")

plt.ylim(0.05, 10000)
plt.yscale("log")
plt.xlabel(r"$r / R_\odot$")
plt.ylabel("$\\nabla$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-check-profile.pgf", format="pgf")
plt.show()
plt.close()

profile.bulk_names
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, profile in enumerate(profiles):
    plt.plot(10**profile.logR, profile.entropy, c=f"C{i}")

plt.ylim(19, 48)
plt.xlabel(r"$r / R_\odot$")
plt.ylabel("$s / (\\textrm{N}_\\textrm{A}\\textrm{k}_\\textrm{b})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-check-profile-2.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)


index = np.argwhere(wind_sal5.R / wind_sal5.rl_1 > 1)[0][0]
plt.xlabel("Time (yr)")
axs[0].set_ylabel("$M_\\textrm{conv}$ ($M_\odot$)")
axs[1].set_ylabel("$M_\\textrm{star} -M_\\textrm{conv}$ ($M_\odot$)")

axs[0].plot(
    wind_sal5.star_age,
    wind_sal5.M_conv,
    c="C0",
)
axs[0].scatter(
    wind_sal5.star_age[index], wind_sal5.M_conv[index], c="C0", label="Start of RLOF"
)
axs[0].set_xlim(axs[0].get_xlim())

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.star_mass - wind_sal5.M_conv,
    c="C0",
    label=r"$\beta_\textrm{Saladino}$",
)
axs[1].scatter(
    wind_sal5.star_age[index],
    wind_sal5.star_mass[index] - wind_sal5.M_conv[index],
    c="C0",
    label="Start of RLOF",
)

axs[0].legend()
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-3.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    2, 1, figsize=set_size(column, height=0.75), constrained_layout=True, sharex=True
)


plt.xlabel("Time (yr)")
axs[0].set_ylabel("$t_\\textrm{conv}$ (months)")
axs[1].set_ylabel("$f_\\textrm{conv}$")

axs[0].plot(
    wind_sal5.star_age,
    wind_sal5.t_conv / 3600 / 24 / 365.25 * 12,
    c="C0",
)
axs[0].scatter(
    wind_sal5.star_age[index],
    wind_sal5.t_conv[index] / 3600 / 24 / 365.25 * 12,
    c="C0",
    label="Start of RLOF",
)

axs[0].set_xlim(axs[0].get_xlim())

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.f_conv,
    c="C0",
    label=r"$\beta_\textrm{Saladino}$",
)

axs[0].legend()
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-4.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    2, 1, figsize=set_size(column, height=0.75), constrained_layout=True, sharex=True
)


plt.xlabel("Time (yr)")
axs[0].set_ylabel("$I$ (g cm$^2$)")
axs[1].set_ylabel("$r_\\textrm{g}^2$")

print(wind_sal5.bulk_names)
axs[0].plot(
    wind_sal5.star_age,
    wind_sal5.I_eff,
    c="C0",
)

axs[0].plot(
    wind_sal5.star_age,
    2.8e59 * wind_sal5.R**2 / wind_sal5.R[0] ** 2,
    c="C9",
    label=r"Scaled $R^2$",
    linewidth=4,
    zorder=-1,
    alpha=0.5,
)
axs[0].scatter(
    wind_sal5.star_age[index],
    wind_sal5.I_eff[index],
    c="C0",
)
axs[0].set_xlim(axs[0].get_xlim())

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.I_eff / (wind_sal5.star_mass * Msun * (wind_sal5.R * Rsun) ** 2),
    c="C0",
)
axs[1].scatter(
    wind_sal5.star_age[index],
    wind_sal5.I_eff[index]
    / (wind_sal5.star_mass[index] * Msun * (wind_sal5.R[index] * Rsun) ** 2),
    c="C0",
    label="Start of RLOF",
)

axs[0].legend()
axs[1].legend()

plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-10.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    2, 1, figsize=set_size(column, height=0.75), constrained_layout=True, sharex=True
)


plt.xlabel("Time (yr)")
axs[0].set_ylabel("$\Omega$ (s$^{-1}$)")
axs[1].set_ylabel("$\Omega$ (s$^{-1}$)")

print(wind_sal5.bulk_names)
axs[0].plot(
    wind_sal5.star_age,
    wind_sal5.Omega_star,
    label=r"Star",
    c="C0",
)

axs[0].plot(
    wind_sal5.star_age,
    wind_sal5.Omega_orb,
    c="C1",
    label=r"Orbit",
    linewidth=3,
    zorder=-1,
    alpha=0.5,
)
axs[0].scatter(
    wind_sal5.star_age[index],
    wind_sal5.Omega_star[index],
    c="C0",
    label="Start of RLOF",
)
axs[0].set_xlim(axs[0].get_xlim())

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.Omega_star,
    c="C0",
)

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.Omega_orb,
    c="C1",
    label=r"Scaled $R^2$",
    linewidth=3,
    zorder=-1,
    alpha=0.5,
)
axs[1].scatter(
    wind_sal5.star_age[index],
    wind_sal5.Omega_star[index],
    c="C0",
)

axs[0].legend(ncols=3)
axs[1].set_ylim(0.25e-7, 0.3e-7)

plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-11.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([wind_sal7]):
    axs.plot(
        model.star_age,
        model.I_eff * model.Omega_star,
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )

    axs.plot(
        model.star_age,
        model.angular_momentum_j_orbit,
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )

    J_loss = -np.cumsum((model.jdot_ml + model.jdot_gr) * model.dt * 365.25 * 24 * 3600)

    axs.plot(
        model.star_age,
        model.angular_momentum_j_orbit + model.I_eff * model.Omega_star + J_loss,
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )

    J_loss_star = np.cumsum(
        (
            10**model.lg_wind_mdot_1
            * Msun
            * (model.R * Rsun) ** 2
            * model.Omega_star
            / 1.5
        )
        * model.dt
    )
    print(J_loss_star)
    J_loss += J_loss_star

    axs.plot(
        model.star_age,
        model.angular_momentum_j_orbit + model.I_eff * model.Omega_star + J_loss,
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )

plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-5.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

model = wind_sal5

axs.plot(
    model.star_age,
    model.lg_wind_mdot_1,
    label=r"$\beta_\textrm{Saladino}$",
)

plt.xlabel("Time (yr)")
plt.ylabel(r"$\log(\dot{M}_\textrm{wind} / (M_\odot / \textrm{ yr}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-20.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([wind_sal5, wind_sal6, wind_sal7]):

    J_loss = -np.cumsum((model.jdot_ml + model.jdot_gr) * model.dt * 365.25 * 24 * 3600)
    J_loss_star = np.cumsum(
        (
            10**model.lg_wind_mdot_1
            * Msun
            * (model.R * Rsun) ** 2
            * model.Omega_star
            / 1.5
        )
        * model.dt
    )
    J_loss += J_loss_star

    axs.plot(
        model.star_age,
        1
        - (model.angular_momentum_j_orbit + model.I_eff * model.Omega_star + J_loss)
        / (model.angular_momentum_j_orbit + model.I_eff * model.Omega_star + J_loss)[0],
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )


plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-5.pgf.pgf", format="pgf")
plt.show()
plt.close()
model.bulk_names

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([wind_sal5, wind_sal6, wind_sal7]):

    J_loss = -np.cumsum((model.jdot_ml + model.jdot_gr) * model.dt * 365.25 * 24 * 3600)
    J_loss_star = np.cumsum(
        (
            10**model.lg_wind_mdot_1
            * Msun
            * (model.R * Rsun) ** 2
            * model.Omega_star
            / 1.5
        )
        * model.dt
    )
    J_loss += J_loss_star

    axs.plot(
        model.star_age,
        model.star_2_mass,
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )


plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-5.pgf.pgf", format="pgf")
plt.show()
plt.close()
model.bulk_names


# %%


def compute_vwind(h):

    G = 6.67430e-8
    Msun = 1.989e33
    Rsun = 6.957e10

    m = h.star_mass * Msun
    R = h.R * Rsun
    log_Teff = np.log10(h.Teff)

    v_esc = np.sqrt(2.0 * G * m / R)

    vw_factor = np.zeros_like(log_Teff)

    # piecewise definition
    mask1 = log_Teff < 3.5
    mask2 = (log_Teff >= 3.5) & (log_Teff <= 4.0)
    mask3 = (log_Teff > 4.0) & (log_Teff <= 4.35)
    mask4 = log_Teff > 4.35

    vw_factor[mask1] = 0.35
    vw_factor[mask2] = 0.7 + 0.7 * (log_Teff[mask2] - 4.0)
    vw_factor[mask3] = 1.3
    vw_factor[mask4] = 2.6

    vwind = vw_factor * v_esc

    return vwind


def compute_vexp(h):

    G = 6.67430e-8
    Msun = 1.989e33
    Rsun = 6.957e10

    logP = -2.07 + 1.94 * np.log10(h.R) - 0.9 * np.log10(h.star_mass)
    P = 10.0**logP

    v_exp = -13.5 + 0.056 * P
    v_exp = [3 if v_val < 3 else v_val for v_val in v_exp]
    v_exp = [15 if v_val > 15 else v_val for v_val in v_exp]
    return v_exp


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(wind_sal5.age, compute_vwind(wind_sal5) / 100000, label="Lamers")
plt.plot(wind_sal5.age, compute_vexp(wind_sal5), label="Vassiliadis + Wood")


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("Time (yr)")
plt.ylabel(r"$v_\infty$ (km s$^{-1}$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-check-vwind.pgf", format="pgf")
plt.show()
plt.close()
# %%


def beta_Sal(q, v):

    c1 = 1.7 + 0.3 / q
    c2 = 0.5 + 0.2 / q
    alpha = 1.0 / (c1 + (c2 * v) ** 5.0) + 0.75
    beta_max = min(0.3, 1.4 * q**2)
    beta = min(beta_BHL(q, v, alpha), beta_max)
    return beta


def beta_BHL(q, v, alpha):
    vrel = np.sqrt(1 + v**2)
    beta = alpha / ((1 + 1 / q) ** 2 * v * vrel**3)
    return beta


def eta_sal(q, v):
    eta_iso = (1 + 1 / q) ** (-2)
    c1 = max(1 / q, 0.6 * (1 / q) ** 1.7)
    c2 = 1.5 + 0.3 / q
    eta = min(1.0 / (c1 + (c2 * v) ** 3) + eta_iso, 0.6)
    return eta


def get_v_orbit(h):
    Mtot = h.star_1_mass + h.star_2_mass
    a = h.binary_separation
    Mtot = Mtot * Msun
    a = a * Rsun
    return np.sqrt(G * Mtot / a)


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

vw_over_vorbits = compute_vexp(wind_sal5) / get_v_orbit(wind_sal5)
vw_over_vorbits *= 100000

qs = wind_sal5.star_2_mass / wind_sal5.star_1_mass
beta = np.array(
    [beta_Sal(qs[i], vw_over_vorbits[i]) for i in range(len(wind_sal5.star_age))]
)
eta = np.array(
    [eta_sal(qs[i], vw_over_vorbits[i]) for i in range(len(wind_sal5.star_age))]
)

(l1,) = plt.plot(wind_sal5.star_age, beta, label=r"$\beta_\textrm{Saladino}$")
(l3,) = plt.plot(wind_sal5.star_age, eta, label=r"$\eta_\textrm{Saladino}$")
(l5,) = plt.plot(
    wind_sal5.star_age, vw_over_vorbits, label=r"$v_\textrm{exp} / v_\textrm{orbit}$"
)
(l6,) = plt.plot(wind_sal5.star_age, qs, label=r"$q = M_a / M_d$")
plt.ylim(0, 1.2)
axs.set_xlim(axs.get_xlim())
l2 = plt.hlines(
    0.3,
    -1e10,
    1e10,
    color="C0",
    linewidth=0.75,
    alpha=0.3,
    label=r"$\beta_\textrm{max}$",
)
l4 = plt.hlines(
    0.6,
    -1e10,
    1e10,
    color="C1",
    linewidth=0.75,
    alpha=0.3,
    label=r"$\eta_\textrm{max}$",
)

fig.legend(loc="outside upper center", ncols=3, handles=[l1, l2, l3, l4, l5, l6])
plt.xlabel("Time (yr)")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-compare-wind.pgf", format="pgf")
plt.show()
plt.close()

# %%

vw_over_vorbits = compute_vexp(wind_sal5) / get_v_orbit(wind_sal5)
vw_over_vorbits *= 100000

plt.plot(wind_sal5.star_age, vw_over_vorbits)
plt.plot(wind_sal5.star_age, wind_sal5.vwind_over_vorbit)


plt.show()
# %%


def mdot(h):
    vexp = np.array(compute_vwind(h))
    logP = -2.07 + 1.94 * np.log10(h.R) - 0.9 * np.log10(h.star_mass)
    P = 10.0**logP

    Mdot1 = 10.0 ** (-11.4 + 0.0123 * P)
    Mdot2 = h.L * 3.828e33 * 3.154e7 / (2.9979e10 * vexp * 1.9885e33)
    return [min(Mdot1[i], Mdot2[i]) for i in range(len(Mdot1))]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

model = wind_sal5

mdots = np.log10(mdot(wind_sal5))
axs.plot(
    model.star_age,
    model.lg_wind_mdot_1,
    label=r"Vassiliadis + Wood",
    alpha=0.7,
    linewidth=5,
    c="C9",
)
axs.plot(model.star_age, mdots, label=r"Lamers", c="C0")


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("Time (yr)")
plt.ylabel(r"$\log(\dot{M}_\textrm{wind} / (M_\odot / \textrm{ yr}))$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-40.pgf", format="pgf")
plt.show()
plt.close()

# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([wind_sal7]):
    axs.plot(
        model.star_age,
        model.I_eff * model.Omega_star,
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )

    axs.plot(
        model.star_age,
        model.angular_momentum_j_orbit,
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )

    J_loss = -np.cumsum((model.jdot_ml + model.jdot_gr) * model.dt * 365.25 * 24 * 3600)

    axs.plot(
        model.star_age,
        model.angular_momentum_j_orbit + model.I_eff * model.Omega_star + J_loss,
        c=f"C{i}",
        label=r"$\beta_\textrm{Saladino}$",
    )

    J_loss_star = np.cumsum(
        (
            10**model.lg_wind_mdot_1
            * Msun
            * (model.R * Rsun) ** 2
            * model.Omega_star
            / 1.5
        )
        * model.dt
    )
plt.xlabel("")
plt.ylabel("")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-50.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

for i, model in enumerate([wind_sal5]):
    axs[0].plot(
        model.star_age,
        model.I_eff * model.Omega_star,
        c=f"C0",
        label=r"Orbit",
    )

    axs[0].plot(
        model.star_age,
        model.angular_momentum_j_orbit,
        c=f"C1",
        label=r"Star",
    )

    axs[0].plot(
        model.star_age,
        model.angular_momentum_j_orbit + model.I_eff * model.Omega_star,
        c=f"C2",
        label=r"Orbit + Star",
    )
    axs[1].plot(
        model.star_age,
        model.angular_momentum_j_orbit + model.I_eff * model.Omega_star,
        c=f"C2",
    )


fig.legend(loc="outside upper center", ncols=3)
axs[1].set_xlabel("Time (yr)")
axs[0].set_ylabel("$J$ (g cm$^2$ s$^{-1}$)")
axs[1].set_ylabel("$J$ (g cm$^2$ s$^{-1}$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-5.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([wind_sal5]):
    J_loss_ml = -np.cumsum((model.jdot_ml) * model.dt * 365.25 * 24 * 3600)
    J_loss_gr = -np.cumsum((model.jdot_gr) * model.dt * 365.25 * 24 * 3600)
    J_loss_star = np.cumsum(
        (
            10**model.lg_wind_mdot_1
            * Msun
            * (model.R * Rsun) ** 2
            * model.Omega_star
            / 1.5
        )
        * model.dt
    )

    axs[0].plot(
        model.star_age,
        J_loss_ml,
        c=f"C0",
        label=r"Orbit wind loss",
    )
    axs[0].plot(
        model.star_age,
        J_loss_star,
        c=f"C1",
        label=r"Star wind loss",
    )
    axs[0].plot(
        model.star_age,
        J_loss_gr,
        c=f"C2",
        label=r"GR",
    )

    axs[1].plot(
        model.star_age,
        model.angular_momentum_j_orbit
        + model.I_eff * model.Omega_star
        + J_loss_star
        + J_loss_ml
        + J_loss_gr,
    )


axs[0].legend(ncols=3)
axs[0].set_yscale("log")
axs[1].set_xlabel("Time (yr)")
axs[0].set_ylabel(r"$J_\textrm{losses}$ (g cm$^2$ s$^{-1}$)")
axs[1].set_ylabel(r"$J_\textrm{system}$ (g cm$^2$ s$^{-1}$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-50.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([wind_sal7]):
    J_loss_ml = -np.cumsum((model.jdot_ml) * model.dt * 365.25 * 24 * 3600)
    J_loss_gr = -np.cumsum((model.jdot_gr) * model.dt * 365.25 * 24 * 3600)
    J_loss_star = np.cumsum(
        (
            10**model.lg_wind_mdot_1
            * Msun
            * (model.R * Rsun) ** 2
            * model.Omega_star
            / 1.5
        )
        * model.dt
    )

    axs[0].plot(
        model.star_age[1:],
        np.diff(model.lg_wind_mdot_1) / model.dt[1:],
        c=f"C2",
    )

    axs[1].plot(
        model.star_age,
        model.angular_momentum_j_orbit
        + model.I_eff * model.Omega_star
        + J_loss_star
        + J_loss_ml
        + J_loss_gr,
    )


axs[0].legend(ncols=3)
axs[1].set_xlabel("Time (yr)")
axs[0].set_ylabel(r"$J_\textrm{losses}$ (g cm$^2$ s$^{-1}$)")
axs[1].set_ylabel(r"$J_\textrm{system}$ (g cm$^2$ s$^{-1}$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-51.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs[0].plot(
    wind_sal5.star_age,
    wind_sal5.rl_1,
    alpha=0.8,
    linewidth=4,
    c="C9",
    label="No accretion",
)
axs[0].plot(wind_sal7.star_age, wind_sal7.rl_1, label="Accretion")
axs[0].set_ylim(420, 530)


# interpolate second onto first
age_ref = wind_sal5.star_age
RL_ref = wind_sal5.rl_1

# second model
age_other = wind_sal7.star_age
RL_other = wind_sal7.rl_1

RL_other_interp = np.interp(age_ref, age_other, RL_other)

# difference
diff = RL_other_interp - RL_ref

axs[1].plot(age_ref, np.log10(np.abs(diff)), c="k")


axs[0].legend()
axs[1].set_xlabel("Time (yr)")
axs[0].set_ylabel(r"$R_\textrm{RL}$ ($R_\odot$)")
axs[1].set_ylabel(r"$\log|$residual $(R_\odot)|$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-accretion-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs[0].plot(
    wind_sal5.star_age,
    wind_sal5.star_2_mass,
    alpha=0.8,
    linewidth=4,
    c="C9",
    label="No accretion",
)
axs[0].plot(wind_sal7.star_age, wind_sal7.star_2_mass, label="Accretion")

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.star_2_mass - wind_sal5.star_2_mass[0],
    alpha=0.8,
    linewidth=4,
    c="C9",
    label="No accretion",
)
axs[1].plot(
    wind_sal7.star_age,
    wind_sal7.star_2_mass - wind_sal7.star_2_mass[0],
    label="Accretion",
)


axs[1].set_yscale("log")

axs[0].legend()
axs[1].set_xlabel("Time (yr)")
axs[0].set_ylabel(r"$M_\textrm{a}$ ($M_\odot$)")
axs[1].set_ylabel(r"$M_\textrm{a} - M_\textrm{a,i}$ ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-accretion-2.pgf", format="pgf")
plt.show()
plt.close()

# %%
