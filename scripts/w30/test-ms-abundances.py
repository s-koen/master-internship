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
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

grid = MesaGrid(f"{MASTER}/grid-masses-2026-08-14-clean")
grid2 = MesaGrid(f"{MASTER}/grid-masses-2-2026-08-16-clean")
grid3 = MesaGrid(f"{MASTER}/grid-masses-3-2026-08-24")
grid4 = MesaGrid(f"{MASTER}/grid-masses-4-2026-08-25")
grid.merge(grid2)
grid.merge(grid3, overwrite=True)
grid.merge(grid4, overwrite=True)

# %%
df = AbundanceTables()


def get_ba(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    ab = Abundances(model=r, df=df, full_mixing=True)
    ab2 = Abundances(model=r, df=df, full_mixing=False)
    return ab.MS_massfrac[52] / ab2.MS_massfrac[52]


ratios = []
ms = [1.8, 2.2, 2.6, 3.0]

for m in ms:
    R, q, ratio = grid.array(get_ba, "R", "q", m=m)
    ratios.append(ratio)


minn = np.nanmin(ratios)
maxx = np.nanmax(ratios)


# %%

fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

axs = axs.flatten()

for i, m in enumerate(ms):
    c = axs[i].pcolormesh(R, q, ratios[i].T, cmap="viridis", vmin=minn, vmax=maxx)
    axs[i].set_title(f"$M_\\textrm{{TPAGB,i}} = {m:.1f}\\;M_\\odot$")
plt.colorbar(
    c,
    ax=axs,
    label="$X(\\textrm{Ba})_\\textrm{MS, full mixing} / X(\\textrm{Ba})_\\textrm{MS, informed mixing}$",
)

axs[0].set_ylabel("$q$")
axs[2].set_ylabel("$q$")
axs[2].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
axs[3].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-large-grid-3-fix.pgf", format="pgf")
plt.show()
plt.close()


# %%

m = grid.models[30]

df = AbundanceTables()
ab = Abundances(model=m, df=df, save_accretor=True)

plt.plot(ab.accretor_res.mass_profile, ab.accretor_res.mu_profile_mix)
plt.plot(ab.accretor_res.mass_profile, ab.accretor_res.mu_profile_original)
plt.axhline(ab.accretor_res.final_mu)
plt.show()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


ab = Abundances(model=m, df=df)
z = ab.elements_mass
plt.plot(
    z,
    ab.MS_spectroscopic,
    linewidth=1,
)

# plt.plot(z, ab.initial_envelope_abundances["massfrac"], c=f"C9", zorder=-10)
element_labels(fig, ab.elements_mass, ab.elements_name)

for z in [38, 40, 56, 58]:
    plt.axvline(z, c="C9", linewidth=0.75 / 2, zorder=-10)


fig.legend(loc="outside upper center", ncols=3)

plt.ylabel("$\\varepsilon_i$")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-test-spec.pgf", format="pgf")
plt.show()
plt.close()
# %%

asplund = Asplund()

# %%

specs = ab.MS_spectroscopic
fe_star = specs[22]

sr_fe_star = specs[34] - fe_star
y_fe_star = specs[35] - fe_star
zr_fe_star = specs[36] - fe_star
ba_fe_star = specs[52] - fe_star
la_fe_star = specs[53] - fe_star
ce_fe_star = specs[54] - fe_star
nd_fe_star = specs[56] - fe_star


fe_sun = asplund.elements[26].abundance
sr_fe_sun = asplund.elements[38].abundance - fe_sun
y_fe_sun = asplund.elements[39].abundance - fe_sun
zr_fe_sun = asplund.elements[40].abundance - fe_sun
ba_fe_sun = asplund.elements[56].abundance - fe_sun
la_fe_sun = asplund.elements[57].abundance - fe_sun
ce_fe_sun = asplund.elements[58].abundance - fe_sun
nd_fe_sun = asplund.elements[60].abundance - fe_sun

s_fe = (
    1
    / 7
    * (
        sr_fe_star / sr_fe_sun
        + y_fe_star / y_fe_sun
        + zr_fe_star / zr_fe_sun
        + ba_fe_star / ba_fe_sun
        + la_fe_star / la_fe_sun
        + ce_fe_star / ce_fe_sun
        + nd_fe_star / nd_fe_sun
    )
)

print(s_fe)
# %%
df = AbundanceTables()


def get_r(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    ab = Abundances(model=r, df=df, full_mixing=True)
    ab2 = Abundances(model=r, df=df, full_mixing=False)
    return ab2.mixing_mass / ab.mixing_mass


ratios = []
ms = [1.8, 2.2, 2.6, 3.0]

for m in ms:
    R, q, ratio = grid.array(get_r, "R", "q", m=m)
    ratios.append(ratio)


minn = np.nanmin(ratios)
maxx = np.nanmax(ratios)


# %%

fig, axs = plt.subplots(
    2, 2, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)

axs = axs.flatten()

for i, m in enumerate(ms):
    c = axs[i].pcolormesh(R, q, ratios[i].T, cmap="viridis", vmin=minn, vmax=maxx)
    axs[i].set_title(f"$M_\\textrm{{TPAGB,i}} = {m:.1f}\\;M_\\odot$")
plt.colorbar(
    c,
    ax=axs,
    label="$X(\\textrm{Ba})_\\textrm{MS, full mixing} / X(\\textrm{Ba})_\\textrm{MS, informed mixing}$",
)

axs[0].set_ylabel("$q$")
axs[2].set_ylabel("$q$")
axs[2].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
axs[3].set_xlabel("$R_\\textrm{RL}$ ($R_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-large-grid-mixing.pgf", format="pgf")
plt.show()
plt.close()
# %%
df = AbundanceTables()


def get_r(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    ab = Abundances(model=r, df=df, full_mixing=True)
    ab2 = Abundances(model=r, df=df, full_mixing=False)
    return ab.MS_massfrac[52] / ab2.MS_massfrac[52]


ratios = []
ms = [1.8, 2.2, 2.6, 3.0]

for m in ms:
    R, q, ratio = grid.array(get_r, "R", "q", m=m)
    ratios.append(ratio)


minn = np.nanmin(ratios)
maxx = np.nanmax(ratios)
