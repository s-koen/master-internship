import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from matplotlib.ticker import ScalarFormatter
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.plot_helpers import *
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)
from scripts.general_utils.accretor import *
from scripts.general_utils.asplund import Element, Asplund

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary

# %%

grid = MesaGrid(f"{MASTER}/grid-broad-R-2026-09-27")
df = AbundanceTables()

grid_sorted = []
rs = []
for model in grid.models:
    rs.append(model.params["R"])
    grid_sorted.append(model)

rs = np.array(rs)
grid_sorted = np.array(grid_sorted)

sort_idx = np.argsort(rs)
grid_sorted = grid_sorted[sort_idx]
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

suc_rs = []
fail_rs = []
suc_ys = []
fail_ys = []

grid_sorted = []
rs = []
for model in grid.models:
    rs.append(model.params["R"])
    grid_sorted.append(model)

rs = np.array(rs)
grid_sorted = np.array(grid_sorted)

sort_idx = np.argsort(rs)
grid_sorted = grid_sorted[sort_idx]

for model in grid_sorted[36:][::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c3 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Blues",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
for model in grid_sorted[:36][::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c2 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Greens",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )

plt.colorbar(c1, label="log$(M_\\textrm{env} / M_\\odot)$", pad=0.01, aspect=50)
cbar = plt.colorbar(c3, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])
cbar = plt.colorbar(c2, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{star}$ ($R_\\odot$)")
plt.ylabel("$\\dot{M} / \\dot{M}_\\textrm{crit}$")
# plt.savefig(
#     "/home/koen/LaTeX-setup/plots/w30-failing-models-6.pgf", format="pgf", dpi=600
# )
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

per = []
s = []
for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue

    per.append(m.params["R"])
    s.append(m.star_2_mass[-1] - m.params["m"] * m.params["q"])

per = np.array(per)
s = np.array(s)
idx = np.argsort(per)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)

plt.axhline(0.25 * m.envelope_mass[0], c="C9", linewidth=0.75, zorder=-10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("$\\Delta M_\\textrm{acc}$ ($M_\\odot$)")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-macc-RL.pgf", format="pgf")
plt.show()
plt.close()
# %%

for model in grid_sorted[5:6]:
    profiles = model.profiles
    for p in profiles:
        plt.plot(
            p.mass[0] - p.mass,
            10**p.log_thermal_time_to_surface / 3600 / 24 / 365,
            c="C0",
        )


for model in grid_sorted[40:41]:
    profiles = model.profiles
    for p in profiles:
        plt.plot(
            p.mass[0] - p.mass,
            10**p.log_thermal_time_to_surface / 3600 / 24 / 365,
            c="C1",
        )


plt.show()


# %%

norm = plt.Normalize(150, 404)
cmap = plt.cm.viridis
# color = cmap(norm(x))


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted[:26][::-1]:
    print(model.params["R"])
    ents = []
    ms = []
    ts = []

    profiles = model.profiles
    for p in profiles:
        args = np.argmax(p.entropy)
        mass = p.mass[:args]
        entropy = p.entropy[:args]
        argmin = np.argmin(entropy)
        ents.append(entropy[argmin])
        ms.append(mass[argmin] - p.he_core_mass)
        # ents.append(p.entropy[0])
        # ms.append(p.mass[0] - p.he_core_mass)
    if model.envelope_mass[-1] > 0.01:
        c = "C3"
        plt.plot(ms, ents, c=c, linewidth=3, alpha=0.5, zorder=-1)
    c = cmap(norm(model.params["R"]))
    plt.plot(ms, ents, c=c)


# for model in grid_sorted[40:41]:
#     profiles = model.profiles
#     for p in profiles:
#         plt.plot(p.mass[0] - p.mass, p.entropy, c="C1")
#

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

plt.xlabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("Minimum entropy in envelope")

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w31-min-entropy.pgf", format="pgf")
plt.show()
plt.close()

# %%

norm = plt.Normalize(150, 404)
cmap = plt.cm.viridis
# color = cmap(norm(x))


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted[:26][::-1]:
    print(model.params["R"])
    ents = []
    ms = []
    ts = []

    profiles = model.profiles
    for p in profiles:
        args = np.argmax(p.entropy)
        mass = p.R[:args]
        entropy = p.entropy[:args]
        argmin = np.argmin(entropy)
        ents.append(entropy[argmin])
        ms.append(mass[argmin])
        # ents.append(p.entropy[0])
        # ms.append(p.mass[0] - p.he_core_mass)
    if model.envelope_mass[-1] > 0.01:
        c = "C3"
        plt.plot(ms, ents, c=c, linewidth=3, alpha=0.5, zorder=-1)
    c = cmap(norm(model.params["R"]))
    plt.plot(ms, ents, c=c)


# for model in grid_sorted[40:41]:
#     profiles = model.profiles
#     for p in profiles:
#         plt.plot(p.mass[0] - p.mass, p.entropy, c="C1")
#

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

plt.xlabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("Minimum entropy in envelope")

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-min-entropy.pgf", format="pgf")
plt.show()
plt.close()


# %%

profiles = grid_sorted[16].profiles[::2]

norm = plt.Normalize(np.log10(0.001), np.log10(1.6))
cmap = plt.cm.viridis
# color = cmap(norm(x))


for p in profiles:
    argmax = np.argmax(p.entropy)
    plt.plot(
        p.mass[0] - p.mass[:argmax],
        p.entropy[:argmax],
        c=cmap(norm(np.log10(p.star_mass - p.he_core_mass))),
    )
    args = np.argwhere(p.gradT[:argmax] - p.grada[:argmax] > 0.1)
    plt.plot(
        p.mass[0] - p.mass[:argmax][args],
        p.entropy[:argmax][args],
        c="C9",
        alpha=0.5,
        linewidth=5,
    )

plt.xscale("log")
plt.ylim(21, 23)
plt.xlim(1e-8)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"label")

plt.show()
print(p.bulk_names)

# %%

profiles = grid_sorted[17].profiles[1:5]
for p in profiles:
    plt.plot(p.R, p.gradT - p.grada)
plt.show()

print(p.bulk_names)


# %%

norm = plt.Normalize(150, 404)
cmap = plt.cm.viridis
# color = cmap(norm(x))


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted[:26][::-1]:
    print(model.params["R"])
    ents = []
    ms = []
    ts = []

    profiles = model.profiles
    for p in profiles:
        args = np.argmax(p.entropy)
        mass = p.R[:args]
        entropy = p.entropy[:args]
        argmin = np.argmin(entropy)
        ents.append(entropy[argmin])
        # ms.append(mass[argmin])
        # ents.append(p.entropy[0])
        ms.append(p.star_age)
    if model.envelope_mass[-1] > 0.01:
        c = "C3"
        plt.plot(ms, ents, c=c, linewidth=3, alpha=0.5, zorder=-1)
    c = cmap(norm(model.params["R"]))
    plt.plot(ms, ents, c=c)


# for model in grid_sorted[40:41]:
#     profiles = model.profiles
#     for p in profiles:
#         plt.plot(p.mass[0] - p.mass, p.entropy, c="C1")
#

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

plt.xlabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("Minimum entropy in envelope")

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-min-entropy.pgf", format="pgf")
plt.show()
plt.close()


# %%

norm = plt.Normalize(150, 404)
cmap = plt.cm.viridis
# color = cmap(norm(x))


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted[:26][::-1]:
    print(model.params["R"])
    ents = []
    ms = []
    ts = []

    profiles = model.profiles
    for p in profiles:
        args = np.argmax(p.entropy)
        mass = p.mass[:args]
        entropy = p.entropy[:args]
        tt = p.thermal_time_to_surface[:args]
        argmin = np.argmin(entropy)
        ents.append(tt[argmin])
        ms.append(mass[argmin])
        # ents.append(p.entropy[0])
        # ms.append(p.star_age)
    if model.envelope_mass[-1] > 0.01:
        c = "C3"
        plt.plot(ms, ents, c=c, linewidth=3, alpha=0.5, zorder=-1)
    c = cmap(norm(model.params["R"]))
    plt.plot(ms, ents, c=c)


# for model in grid_sorted[40:41]:
#     profiles = model.profiles
#     for p in profiles:
#         plt.plot(p.mass[0] - p.mass, p.entropy, c="C1")
#

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

plt.xlabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("Minimum entropy in envelope")

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"Initial Roche lobe radius ($R_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-min-entropy.pgf", format="pgf")
plt.show()
plt.close()


# %%
def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted:
    if model.envelope_mass[-1] > 0.01:
        plt.plot(model.envelope_mass, model.R / rol(model), c="C3", linewidth=0.75)
    else:
        plt.plot(model.envelope_mass, model.R / rol(model), c="C2", linewidth=0.75)


plt.axhline(1, c="C9", linewidth=0.75, zorder=-1)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("$R_\\textrm{star} / R_\\textrm{outer lobe}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w31-ROL.pgf", format="pgf")
plt.show()
plt.close()
# %%


def radial_response(history):

    argument_of_RLOF = np.argwhere(history.R / history.rl_1 > 1)[0][0]
    r0 = history.R[argument_of_RLOF]
    mdot = 10 ** history.lg_mstar_dot_1[argument_of_RLOF:]
    ts = history.star_age[argument_of_RLOF:] - history.star_age[argument_of_RLOF]
    dts = np.diff(ts)

    v0 = 4 / 3 * np.pi * r0**3
    m0 = history.star_mass[argument_of_RLOF]
    rho0 = m0 / v0

    rs = history.R[argument_of_RLOF:]

    return ts, r0 + mdot / (4 * np.pi * r0**2 * rho0) * ts, rs


ts, r, rr = radial_response(grid_sorted[14])

plt.plot(ts, r)
plt.plot(ts, rr)
plt.show()
# %%
model.bulk_names
# %%

norm = plt.Normalize(np.log10(150), np.log10(1000))
cmap = plt.cm.viridis
# color = cmap(norm(x))


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for model in grid_sorted[:-2][::-1]:
    print(model.params["R"])
    xs = []
    ys = []

    profiles = model.profiles
    for p in profiles:

        k_max = 0
        k_min = 99999999
        for k in p.zone:
            k = k - 1
            if p.gradT[k] - p.grada[k] > 0.1:
                if k > k_max:
                    k_max = k
                if k < k_min:
                    k_min = k

        dm = (p.mass[k_min] - p.mass[k_max]) / (p.mass[0] - p.he_core_mass)
        xs.append(p.mass[0] - p.he_core_mass)
        ys.append(dm)

    if model.envelope_mass[-1] > 0.01:
        c = "C3"
        plt.plot(xs, ys, c=c, linewidth=3, alpha=0.5, zorder=-1)
    c = cmap(norm(np.log10(model.params["R"])))
    plt.plot(xs, ys, c=c)


# for model in grid_sorted[40:41]:
#     profiles = model.profiles
#     for p in profiles:
#         plt.plot(p.mass[0] - p.mass, p.entropy, c="C1")
#

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

plt.xlabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.ylabel("$M_\\textrm{sad} / M_\\textrm{env}$")
plt.axhline(1, c="C9", linewidth=0.75)

plt.yscale("log")
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"log($R_\textrm{RL} / R_\odot$)")


axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w31-dm-sad.pgf", format="pgf")
plt.show()
plt.close()

# %%
