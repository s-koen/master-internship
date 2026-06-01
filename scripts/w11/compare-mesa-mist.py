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

sys.path.insert(1, "/home/koen/master-internship/scripts/evolve/")
import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid

from scripts.evolve.constants import *
from scripts.evolve.bin_input import *
from scripts.evolve.read_mist_models import *
from scripts.evolve.mrenv import *
from scripts.evolve.orbit_evol import *
from scripts.evolve.rgbf import *
from scripts.evolve.star_model import *

import pickle as pkl

Rsun = 6.957e10
Msun = 1.3271244e26 / 6.67430e-8
G = 6.67430e-8


# %%

with open(
    "/home/koen/master-internship/scripts/evolve_mesa/complete-test.pkl", "rb"
) as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


# %%
n = [0]
for phase in ["MS", "GB", "CHeB", "EAGB", "TPAGB"]:
    data = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/standard-2msun-v2/LOGS/{phase}/history.data"
    )
    n.append(n[-1] + len(data.model_number[1:]))

combined_star = {}
for bulk_name in data.bulk_names:
    combined_star[bulk_name] = np.empty(n[-1])
combined_star["phase"] = np.empty(n[-1])

for i, phase in enumerate(["MS", "GB", "CHeB", "EAGB", "TPAGB"]):
    data = mr.MesaData(
        f"/home/koen/master-internship/mesa-models/standard-2msun-v2/LOGS/{phase}/history.data"
    )
    for bulk_name in data.bulk_names:
        if bulk_name in ["model_number", "star_age"] and i != 0:
            combined_star[bulk_name][n[i] : n[i + 1]] = (
                data.data(bulk_name)[1:] + combined_star[bulk_name][n[i] - 1]
            )
        else:
            combined_star[bulk_name][n[i] : n[i + 1]] = data.data(bulk_name)[1:]
    combined_star["phase"][n[i] : n[i + 1]] = i * np.ones(n[i + 1] - n[i])

# %%

hist = mr.MesaData(f"{MASTER}/wind/9/LOGS/TPAGB/history.data")

# %%

hist2 = mr.MesaData(f"{MASTER}/wind/10/LOGS/TPAGB/history.data")

# %%

delta_t = 49500 + 1.2473e9

# %%
with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
axs.plot(evolve_bin.age, RL_0)
axs.plot(evolve_star.age, 10**evolve_star.log_R)
axs.plot(hist.age + delta_t, hist.R)
axs.plot(hist.age + delta_t, hist.rl_1)
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

print(combined_star.keys())
q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
axs.plot(evolve_bin.age, evolve_bin.spin1)
axs.plot(hist.data("star_age") + delta_t, hist.data("Omega_star"))
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

q = evolve_bin.m2 / evolve_bin.m1
RL_0 = roche_lobe(q) * evolve_bin.a
axs.plot(evolve_bin.age, q)
axs.plot(hist.age + delta_t, hist.star_2_mass / hist.star_1_mass)
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()

# %%
from scipy.interpolate import interp1d

# interpolate log_R onto evolve_bin.age
interp_logR = interp1d(
    evolve_star.age, evolve_star.log_R, bounds_error=False, fill_value="extrapolate"
)

# evaluate on binary age grid
logR_interp = interp_logR(evolve_bin.age)

# compute and plot

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs.plot(evolve_bin.age, (10**logR_interp / evolve_bin.a) ** 6)
axs.plot(hist.age + delta_t, (hist.R / hist.binary_separation) ** 6)
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs.plot(evolve_bin.age[:-1], evolve_bin.beta)
axs.plot(hist.age + delta_t, hist.beta_accretion)
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()


# %%
with open("/home/koen/master-internship/scripts/evolve_mesa/.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs.plot(evolve_bin.age[:-1], evolve_bin.vw_over_vorb)
axs.plot(hist.age + delta_t, hist.vwind_over_vorbit)
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()


# %%

with open("/home/koen/master-internship/scripts/evolve_mesa/.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

axs.plot(evolve_bin.age, evolve_bin.spin1)
axs.plot(hist.age + delta_t, hist.Omega_star)
axs.plot(hist.age + delta_t, hist.Omega_orb)
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()


# %%

fig, axs = plt.subplots(
    3, 1, sharex=True, figsize=set_size(column, height=1.5), constrained_layout=True
)

phases = ["MS", "RGB", "CHeb", "EAGB", "TPAGB"]
axs[0].plot(combined_star["M_conv"], alpha=0.7, linewidth=5, c="C9", label="Reference")
for i in range(5):
    domain = np.where(evolve_star.phase == i)
    axs[0].plot(evolve_star.model[domain], evolve_star.m_cenv[domain], label=phases[i])

axs[1].plot(combined_star["R_conv"], alpha=0.7, linewidth=5, c="C9")
for i in range(5):
    domain = np.where(evolve_star.phase == i)
    axs[1].plot(evolve_star.model[domain], evolve_star.r_cenv[domain])

axs[2].plot(
    combined_star["t_conv"] / 3600 / 24 / 365.25, alpha=0.7, linewidth=5, c="C9"
)
for i in range(5):
    domain = np.where(evolve_star.phase == i)
    axs[2].plot(evolve_star.model[domain], evolve_star.tconv[domain])


axs[0].legend(frameon=False)
axs[0].set_ylabel(r"$M_\textrm{conv}$ ($M_\odot$)")
axs[1].set_ylabel(r"$R_\textrm{conv}$ ($R_\odot$)")
axs[2].set_ylabel(r"$t_\textrm{conv}$ (yr)")
axs[1].set_yscale("log")
axs[2].set_xlabel("Model number")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-conv.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

phases = ["MS", "RGB", "CHeb", "EAGB", "TPAGB"]
axs.plot(combined_star["rg2"], alpha=0.7, linewidth=5, c="C9", label="Reference")
for i in range(5):
    domain = np.where(evolve_star.phase == i)
    axs.plot(evolve_star.model[domain], evolve_star.rg2[domain], label=phases[i])


axs.legend(frameon=False)
axs.set_ylabel(r"$r_\textrm{g}^2$")
axs.set_xlabel("Model number")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-inertia.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

phases = ["MS", "RGB", "CHeb", "EAGB", "TPAGB"]
for i in range(5):
    domain = np.where(evolve_star.phase == i)[0][0]
    time = evolve_star.age[domain]
    domain = np.where(evolve_bin.age > time)
    axs.plot(
        np.arange(0, len(evolve_bin.a))[domain],
        np.log10(1 - evolve_bin.a[domain] / evolve_bin.a[0]),
        label=phases[i],
    )


q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a

axs.legend(frameon=False)
axs.set_ylabel(r"$\log(1 - a / a_i)$")
axs.set_ylim(np.log10(5e-5), np.log10(0.2))
axs.set_xlabel("Model number")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-orbit.pgf", format="pgf"
)
plt.show()
plt.close()


# %%

with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]

with open(
    "/home/koen/master-internship/scripts/evolve_mesa/correct2.pkl", "rb"
) as file:
    rees = pkl.load(file)
    evolve_star2 = rees[0]
    evolve_bin2 = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
rnge = np.argwhere(evolve_bin.age < 1.24762e9 + 2000)
axs.plot(evolve_bin.age[rnge], RL_0[rnge])

q = evolve_bin2.m1 / evolve_bin2.m2
RL_0 = roche_lobe(q) * evolve_bin2.a
rnge = np.argwhere(evolve_bin.age > 1.24762e9 + 1500)
axs.plot(evolve_bin2.age[rnge - 1], RL_0[rnge - 1])


axs.plot(hist3.age + delta_t, hist3.rl_1)
axs.plot(hist3.age + delta_t, hist3.R)
plt.xlim(delta_t - 100000, 1.2479e9)
plt.ylim(400, 550)
plt.xlabel("Star age (yr)")
plt.ylabel("Roche lobe Radius ($R_\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-radius-3.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

hist3 = mr.MesaData(f"{MASTER}/wind/11/LOGS/TPAGB/history.data")

# %%

with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

print(hist.bulk_names)
q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
axs.plot(hist.age + delta_t, hist.J_orb)
axs.plot(hist2.age + delta_t, hist2.J_orb)
axs.plot(hist3.age + delta_t, hist3.J_orb)
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()

# %%
with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

print(hist.bulk_names)
q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
axs.plot(hist.age + delta_t, hist.Omega_star)
axs.plot(hist.age + delta_t, hist.Omega_orb)
axs.plot(hist2.age + delta_t, hist2.Omega_star)
axs.plot(hist2.age + delta_t, hist2.Omega_orb)
axs.plot(hist3.age + delta_t, hist3.Omega_star)
axs.plot(hist3.age + delta_t, hist3.Omega_orb)
plt.xlim(delta_t - 50000, 1.25e9)
plt.xlabel("")
plt.ylabel("")
plt.show()


# %%

plt.plot(combined_star["star_age"], combined_star["rg2"])
plt.plot(hist2.age + delta_t, hist2.rg2)
plt.show()
# %%

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for i, model in enumerate([hist, hist2, hist3]):
    J_loss_ml = -np.cumsum((model.jdot_ml) * model.dt * 365.25 * 24 * 3600)
    J_loss_gr = -np.cumsum((model.jdot_gr) * model.dt * 365.25 * 24 * 3600)
    J_loss_star = np.cumsum(
        (
            10**model.lg_wind_mdot_1
            / (1 - model.beta_accretion)
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
plt.show()
plt.close()


# %%

plt.plot(evolve_bin.age[1:], evolve_bin.vw_over_vorb)
plt.plot(hist3.age + delta_t, hist3.vwind_over_vorbit)
plt.show()
# %%
plt.plot(evolve_star.age, 10**evolve_star.log_R)
plt.plot(hist3.age + delta_t, hist3.R)
plt.show()

# %%
plt.plot(evolve_bin.age, evolve_bin.spin1)
plt.plot(hist3.age + delta_t, hist3.Omega_star)
plt.show()

# %%
with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]

with open(
    "/home/koen/master-internship/scripts/evolve_mesa/correct2.pkl", "rb"
) as file:
    rees = pkl.load(file)
    evolve_star2 = rees[0]
    evolve_bin2 = rees[-1][0]

fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

axs[0].plot(
    evolve_bin.age,
    evolve_bin.m2 / evolve_bin.m1,
    alpha=0.7,
    c="C9",
    linewidth=4,
    label=r"\texttt{evolve.py}",
)
# axs[0].plot(evolve_bin2.age, evolve_bin2.m2 / evolve_bin2.m1)
axs[0].plot(
    hist3.age + delta_t, hist3.star_2_mass / hist3.star_1_mass, label=r"\texttt{MESA}"
)
axs[0].legend()
axs[1].set_xlim(delta_t - 50000, 1.2478e9)
axs[0].set_ylim(0.795, 0.85)


axs[1].plot(
    evolve_star.age[1:],
    np.log10(-evolve_star.dM_dt),
    alpha=0.7,
    c="C9",
    linewidth=5,
    label=r"\texttt{evolve.py}",
)
axs[1].plot(hist3.age + delta_t, hist3.log_abs_mdot, label=r"\texttt{MESA}")
axs[1].set_ylabel(r"$\log(\dot{M} / (M_\odot \textrm{yr}^{-1}))$")
axs[1].set_xlabel(r"Star age (yr)")
axs[0].set_ylabel(r"$q=M_\textrm{a} / M_\textrm{d}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-mdot.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


plt.plot(evolve_star.age[1:], evolve_star.dM_dt)
plt.plot(hist3.age + delta_t, -(10**hist3.log_abs_mdot))
plt.xlim(delta_t - 50000, 1.248e9)
plt.show()

# %%

with open("/home/koen/master-internship/scripts/evolve_mesa/.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
axs.plot(
    evolve_bin.age, RL_0, alpha=0.7, c="C9", linewidth=4, label=r"\texttt{evolve.py}"
)


axs.plot(hist.age + delta_t, hist.rl_1, label=r"\texttt{MESA}")
plt.xlim(delta_t - 100000, 1.2478e9)
plt.ylim(400, 550)
plt.xlabel("Star age (yr)")
plt.ylabel("Roche lobe Radius ($R_\odot$)")
plt.legend()
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-radius-2.pgf", format="pgf"
)
plt.show()
plt.close()

# %%


with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
axs.plot(
    evolve_bin.age, RL_0, alpha=0.7, c="C9", linewidth=4, label=r"\texttt{evolve.py}"
)


axs.plot(hist.age + delta_t, hist.rl_1, label=r"\texttt{MESA}")
plt.xlim(delta_t - 100000, 1.2478e9)
plt.ylim(400, 550)
plt.xlabel("Star age (yr)")
plt.ylabel("Roche lobe Radius ($R_\odot$)")
plt.legend()
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-radius.pgf", format="pgf"
)
plt.show()
plt.close()


# %%
from scipy.interpolate import interp1d

delta_t = 49050 + 1.2473e9

with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]


fig, axs = plt.subplots(
    2, 1, sharex=True, figsize=set_size(column, height=1), constrained_layout=True
)

q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
axs[0].plot(
    evolve_bin.age, RL_0, alpha=0.7, c="C9", linewidth=4, label=r"\texttt{evolve.py}"
)

axs[0].plot(hist3.age + delta_t, hist3.rl_1, label=r"\texttt{MESA}")


# original arrays
x1 = evolve_bin.age
y1 = RL_0

x2 = hist3.age + delta_t
y2 = hist3.rl_1

# make sure mesa is sorted (interp1d requires this)
idx = np.argsort(x2)
x2_sorted = x2[idx]
y2_sorted = y2[idx]

# interpolation function (no extrapolation outside range)
f_interp = interp1d(x2_sorted, y2_sorted, bounds_error=False, fill_value=np.nan)

# evaluate mesa on evolve grid
y2_interp = f_interp(x1)

# residual
residual = y1 - y2_interp

# plot
axs[1].plot(
    x1,
    residual,
)

plt.xlim(delta_t - 100000, 1.2478e9)
axs[1].set_xlabel("Star age (yr)")
axs[0].set_ylabel(r"Roche lobe Radius ($R_\odot$)")
axs[1].set_ylabel(r"Residual ($R_\odot$)")
axs[0].legend()
plt.legend()
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-radius-3.pgf", format="pgf"
)
plt.show()
plt.close()

# %%

with open("/home/koen/master-internship/scripts/evolve_mesa/correct.pkl", "rb") as file:
    rees = pkl.load(file)
    evolve_star = rees[0]
    evolve_bin = rees[-1][0]

with open(
    "/home/koen/master-internship/scripts/evolve_mesa/correct2.pkl", "rb"
) as file:
    rees = pkl.load(file)
    evolve_star2 = rees[0]
    evolve_bin2 = rees[-1][0]


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

q = evolve_bin.m1 / evolve_bin.m2
RL_0 = roche_lobe(q) * evolve_bin.a
rnge = np.argwhere(evolve_bin.age < 1.24762e9 + 2000)
axs.plot(evolve_bin.age[rnge], RL_0[rnge])

q = evolve_bin2.m1 / evolve_bin2.m2
RL_0 = roche_lobe(q) * evolve_bin2.a
rnge = np.argwhere(evolve_bin.age > 1.24762e9 + 1500)
axs.plot(evolve_bin2.age[rnge - 1], RL_0[rnge - 1])


axs.plot(hist3.age + delta_t, hist3.rl_1)
axs.plot(hist3.age + delta_t, hist3.R)
plt.xlim(delta_t - 100000, 1.2479e9)
plt.ylim(400, 550)
plt.xlabel("Star age (yr)")
plt.ylabel("Roche lobe Radius ($R_\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w11-compare-mesa-evolve-radius-3.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
