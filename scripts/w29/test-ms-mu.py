import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)

plt.cplot = cplot

# %%

for m in np.arange(1, 3, 0.1):
    star = get_star(m=m)
    plt.plot(star.age, star.mu)

plt.show()
# %%

data = mr.MesaData(
    "/home/koen/master-internship/mesa-models/single-stars/z0.00557/completed/M1.0/LOGS/MS/profile1.data"
)
data.bulk_names
# %%

print(np.logspace(-4, np.log10(0.7), 50))
# %%

profiles = []
for i in range(1, 41):
    profiles.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/single-ms-stars/M1.9/LOGS/MS/profile{i}.data"
        )
    )
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(0.0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))


print(profile.header_names)

for profile in profiles:
    axs.plot(
        -1 * (profile.mass - profile.mass[0]),
        profile.mu,
        c=cmap(norm(profile.center_h1)),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")
axs.text(0.05, 0.95, "$M=1.9\\;M_\\odot$", transform=axs.transAxes)

axs.set_xscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{star} - m$ ($M_\\odot$)")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-mu-env-1.pgf", format="pgf")
plt.show()
plt.close()

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(0.0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))


print(profile.header_names)

for profile in profiles:
    axs.plot(profile.mass, profile.mu, c=cmap(norm(profile.center_h1)))

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")
axs.text(0.95, 0.95, "$M=1.9\\;M_\\odot$", transform=axs.transAxes, ha="right")

# axs.set_xscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$m$ ($M_\\odot$)")
plt.ylabel("$\\mu$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-mu-core-1.pgf", format="pgf")
plt.show()
plt.close()


# %%
profiles_profiles = []

for i, ax in enumerate(axs):
    print(i)
    profiles = []
    for j in range(1, 41):
        profiles.append(
            mr.MesaData(
                f"/home/koen/master-internship/mesa-models/single-ms-stars/M{0.8+0.1*i:.1f}/LOGS/MS/profile{j}.data"
            )
        )
    profiles_profiles.append(profiles)
# %%

fig, axs = plt.subplots(
    6,
    6,
    sharex=True,
    sharey=True,
    figsize=set_size(full, height=1),
    constrained_layout=True,
)

norm = plt.Normalize(0.0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))

axs = axs.flatten()
for i, ax in enumerate(axs):

    for profile in profiles_profiles[i]:
        ax.plot(
            profile.mass, profile.mu, c=cmap(norm(profile.center_h1)), rasterized=True
        )
    ax.set_title(f"$M={0.8+0.1*i:.1f}\\;M_\\odot$")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=axs, aspect=50)
cbar.set_label(r"$X(\textrm{H})_\textrm{center}$")

ax.set_ylim(0.598, 0.64)
ax.set_xlim(0, 1.5)

# axs.set_xscale("log")
for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)
fig.supxlabel("$M_\\textrm{star} - m$ ($M_\\odot$)", fontsize=10)
fig.supylabel("$\\mu$", fontsize=10)
plt.savefig("/home/koen/LaTeX-setup/plots/w29-mu-core-2.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

import periodictable as pt

# %%
for Z in range(1, 10):
    element = pt.elements[Z]
    print(element.symbol, element.mass)

# %%
with open("data/tp_info_pd_df.pkl", "rb") as f:
    tp_info = pickle.load(f)

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)
# %%

df = AbundanceTables()

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


ab = Abundances(None, df, method="tp offset", mass=2.5)
star = get_star(m=2.5)
final = []
initial = []
z = []
els = []
final.append(ab.h.envelope[-1])
initial.append(ab.h.envelope[0])
final.append(ab.he.envelope[-1])
initial.append(ab.he.envelope[0])
z.append(1)
z.append(2)
els.append("h")
els.append("he")
for el in list(ab.df.elements)[7:]:
    if el in ["tc", "pm", "po"]:
        continue
    e = ab.__getattr__(ab.df.elements[el].key).envelope
    final.append(e[-1])
    initial.append(e[0])
    data = ab.df.envelope[ab.df.envelope["element"] == el]
    z.append(int(np.array(data["elemental_mass"])[0]))
    # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
    els.append(el)

final = final / np.sum(final)
initial = initial / np.sum(initial)

plt.plot(
    z,
    np.array(final),
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)

plt.plot(
    z,
    np.array(initial),
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)

ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)


fig.legend(loc="outside upper center", ncols=3)
plt.xlabel("Element")
plt.ylabel("$X_\\textrm{f} / X_\\textrm{f, Monash $M=2.5\\;M_\\odot,\\;Z=0.007$}$")
# plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
plt.yscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w28-envelope-diff.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

mu_inv = []

for z_i, x_i in zip(z, final):
    element = pt.elements[z_i]
    mu_inv.append(x_i * (1 + z_i) / element.mass)

plt.plot(1 / np.cumsum(mu_inv), label="Final envelope abundances")

mu_inv = []
for z_i, x_i in zip(z, initial):
    element = pt.elements[z_i]
    mu_inv.append(x_i * (1 + z_i) / element.mass)

plt.plot(1 / np.cumsum(mu_inv), label="Initial envelope abundances")

fig.legend(loc="outside upper center", ncols=2)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$Z$")
plt.ylabel("$\sum_{i}1/\mu_i$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-mu-approx-1.pgf", format="pgf")
plt.show()
plt.close()

# %%
import numpy as np


def effective_mu_vs_depth(mass, mu, M_acc, mu_acc):
    """
    Calculate the effective mean molecular weight for every
    possible thermohaline mixing depth.

    Parameters
    ----------
    mass : array_like
        Mass coordinate, ordered from surface -> interior.
    mu : array_like
        Original MS-star mean molecular weight profile.
    M_acc : float
        Accreted mass.
    mu_acc : float
        Mean molecular weight of accreted material.

    Returns
    -------
    m_mix : ndarray
        Mixing depth in mass coordinate.
    mu_eff : ndarray
        Effective mu after mixing to each depth.
    """
    mass = np.asarray(mass)
    mu = np.asarray(mu)

    M_MS = mass[0]

    # dm is negative because the profile goes surface -> interior.
    # Cumulative integral of dm / mu, starting at the surface.
    dm = np.diff(mass)

    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (1.0 / mu[:-1] + 1.0 / mu[1:]) * (-dm))]
    )

    # At each point, cumulative is
    # integral_m^M_MS dm / mu
    M_ms_mixed = M_MS - mass

    mu_eff = (M_acc + M_ms_mixed) / (M_acc / mu_acc + cumulative)

    return mass, mu_eff  # %%


profiles = []
for i in range(1, 41):
    profiles.append(
        mr.MesaData(
            f"/home/koen/master-internship/mesa-models/single-ms-stars/M1.9/LOGS/MS/profile{i}.data"
        )
    )

# %%
M_acc = ab.total_mass_expelled * 0.5

mu_inv = []

for z_i, x_i in zip(z, final):
    element = pt.elements[z_i]
    mu_inv.append(x_i * (1 + z_i) / element.mass)

mu_acc = 1 / np.cumsum(mu_inv)

mass_eff, mu_eff = effective_mu_vs_depth(
    profiles[-1].mass,
    profiles[-1].mu,
    M_acc=0.5,
    mu_acc=0.64,
)

plt.plot(mass_eff, mu_eff)
plt.plot(profiles[-1].mass, profiles[-1].mu)
plt.show()
# %%
