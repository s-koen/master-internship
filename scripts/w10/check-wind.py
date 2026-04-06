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
plt.plot(wind_sal.age, wind_sal.rl_1)
plt.plot(wind_sal2.age, wind_sal2.rl_1)
plt.plot(wind_sal3.age, wind_sal3.rl_1)
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
plt.plot(wind_sal6.age, wind_sal6.rl_1)
plt.plot(wind_sal6.age, wind_sal6.R)
plt.show()
# %%

plt.plot(wind_sal3.star_mass, wind_sal3.period_days)
plt.show()
# %%
wind_sal5.bulk_names
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

plt.xlabel("Time (yr)")

plt.plot(
    wind_sal5.star_age,
    wind_sal5.beta_accretion,
    c="C0",
    label=r"$\beta_\textrm{Saladino}$",
)
axs.set_xlim(axs.get_xlim())
plt.hlines(
    0.3, -1e10, 1e10, color="C0", linewidth=5, alpha=0.3, label=r"$\beta_\textrm{max}$"
)
plt.plot(
    wind_sal5.star_age,
    wind_sal5.eta_accretion,
    c="C1",
    label=r"$\eta_\textrm{Saladino}$",
)
plt.hlines(
    0.6, -1e10, 1e10, color="C1", linewidth=5, alpha=0.3, label=r"$\eta_\textrm{max}$"
)
plt.plot(
    wind_sal5.star_age,
    wind_sal5.vwind_over_vorbit,
    c="C2",
    label=r"$v_\textrm{wind} / v_\textrm{orbit}$",
)

fig.legend(loc="outside upper center", ncols=3)
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

plt.xlabel("Time (yr)")
axs[0].set_ylabel("$M_\\textrm{conv}$ ($M_\odot$)")
axs[1].set_ylabel("$M_\\textrm{star} -M_\\textrm{conv}$ ($M_\odot$)")

axs[0].plot(
    wind_sal5.star_age, wind_sal5.M_conv, c="C0", label=r"$\beta_\textrm{Saladino}$"
)
axs[0].set_xlim(axs[0].get_xlim())

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.star_mass - wind_sal5.M_conv,
    c="C0",
    label=r"$\beta_\textrm{Saladino}$",
)

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
    label=r"$\beta_\textrm{Saladino}$",
)
axs[0].set_xlim(axs[0].get_xlim())

axs[1].plot(
    wind_sal5.star_age,
    wind_sal5.f_conv,
    c="C0",
    label=r"$\beta_\textrm{Saladino}$",
)

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
)

axs[0].legend()

plt.savefig("/home/koen/LaTeX-setup/plots/w10-checks-10.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([wind_sal5, wind_sal6]):
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
model.bulk_names
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([wind_sal5, wind_sal6]):

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
