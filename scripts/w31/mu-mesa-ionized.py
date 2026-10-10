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

profiledict = {}
masses = np.arange(0.4, 4.41, 0.1)
for mass in masses:
    mass = np.round(mass, 1)
    print(mass)
    profiledict[mass] = []
    for i in range(1, 100):
        try:
            prof = mr.MesaData(
                f"/home/koen/master-internship/mesa-models/single-ms-stars-3/M{mass:.1f}/LOGS/MS/profile{i}.data"
            )
            profiledict[mass].append(prof)
        except FileNotFoundError:
            print("oops!", i)
            break

with open("/home/koen/master-internship/scripts/w31/profiles.pkl", "wb") as file:
    pickle.dump(profiledict, file)

# %%
with open("/home/koen/master-internship/scripts/w31/profiles.pkl", "rb") as file:
    profiledict = pickle.load(file)


# %%
def compute_mu_profile(profile):
    names = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"]
    z_s = [1, 2, 2, 6, 7, 8, 10, 12]
    a_s = [1, 3, 4, 12, 14, 16, 20, 24]

    mu_inv = 0
    X_tot = np.zeros(len(profile.mass))
    for n, Z_i, A_i in zip(names, z_s, a_s):
        X_i = profile.data(n)
        mu_inv += X_i * (1 + Z_i) / A_i
        X_tot += X_i
    mu = 1 / mu_inv
    return mu


def compute_mu_asplund(asplund):

    mu_inv = 0
    for element in asplund.elements:
        X_i = asplund.elements[element].massfrac
        Z_i = asplund.elements[element].Z
        A_i = asplund.elements[element].atomic_mass
        mu_inv += X_i * (1 + Z_i) / A_i
    mu = 1 / mu_inv
    return mu


fig, axs = plt.subplots(
    7, 6, sharey=True, figsize=set_size(full, height=1.5), constrained_layout=True
)
axs = axs.flatten()

asplund = Asplund(he_method="karakas", z=0.005573)
mu = compute_mu_asplund(asplund)

norm = plt.Normalize(0, 0.7)
cmap = plt.cm.viridis
# color = cmap(norm(x))

for i, mass in enumerate(masses):
    if i == 41:
        axs[i].axis("off")
        break
    mass = np.round(mass, 1)
    profiles = profiledict[mass]
    for profile in profiles:
        axs[i].plot(
            profile.mass,
            compute_mu_profile(profile),
            c=cmap(norm(profile.center_h1)),
            rasterized=True,
        )
        axs[i].plot(
            profile.mass,
            profile.mu,
            c="C9",
            linewidth=0.75,
            zorder=-10,
            rasterized=True,
        )

    axs[i].set_ylim(0.5975, 0.65)
    axs[i].axhline(mu, c="k", linewidth=0.75, zorder=-1, linestyle=":")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(
    sm,
    ax=axs,
    orientation="horizontal",
    location="top",
    aspect=50,
    label="$X(\\textrm{H})_\\textrm{center}$",
)


for ax in axs:
    ax.spines[["right", "top"]].set_visible(False)

for ax in axs[len(masses) :]:
    ax.remove()
fig.supylabel("$\\mu$", size=10)
fig.supxlabel("$m$ ($M_\\odot$)", size=10)
# plt.savefig("/home/koen/LaTeX-setup/plots/w31-computed-mu.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%
names = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"]
z_s = [1, 2, 2, 6, 7, 8, 10, 12]
a_s = [1, 3, 4, 12, 14, 16, 20, 24]

mu_inv = np.zeros(len(profile.mass))
X_tot = np.zeros(len(profile.mass))

for n, Z, A in zip(names, z_s, a_s):
    X = profile.data(n)
    mu_inv += X * (1 + Z) / A
    X_tot += X

mu_8 = 1 / mu_inv
mu_abar = profile.abar / (1 + profile.zbar)

print("X_tot:")
print(X_tot[::100])

print("\nmu_8 - MESA:")
print((mu_8 - profile.mu)[::100])

print("\nmu_abar - MESA:")
print((mu_abar - profile.mu)[::100])
# %%
