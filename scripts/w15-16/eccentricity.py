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
sys.path.insert(1, "/home/koen/master-internship/scripts/evolve_mesa/")

from bin_input import *
from constants import *
from grid_call import *
from mrenv import *
from orbit_evol import *
from read_mist_models import *
from rgbf import *
from star_model import *
from testev import *
from trho import *

import pandas as pd

MASTER = "/home/koen/master-internship/mesa-models/"


def call_evolution(
    Star, q, a_init, simple_only=False, eps=0.01, f_conv="BSE", e_init=0
):
    # Initial values of the mass ratio, semi-major axis and eccentricity
    # for which the evolution is to be computed

    q_init = [q]
    a_init = [a_init]
    e_init = [e_init]

    # Set modelling options and parameters
    Options = Empty()
    Options.zams_spin_factor = 0.3
    Options.MB_scale_factor = 0

    Options.tide_model = "Preece"  # None / "BSE" / "Preece"
    Options.tidal_freq_scaling = "BSE"  # "BSE" / "Zahn" / "Duguid"
    Options.tide_max_fconv = 1e6
    Options.tide_scale_factor = 1.0
    Options.simple_only = simple_only
    Options.eps = eps
    Options.tidal_freq_scaling = f_conv

    Bins = []
    # Compute and save the orbital evolution for the above system(s)
    for i, (q0, a0, e0) in enumerate(zip(q_init, a_init, e_init)):

        # Compute orbital evolution
        Bin = evolve_orbit(Star, q0, a0, e0, Options, solve_method="LSODA", verbosity=1)

        Bins.append(Bin)

    return [Star, Options, q_init, a_init, e_init, Bins]


# %%

standard_dir = f"{MASTER}/standard-2msun-v3/"
Star = read_stellar_models(standard_dir)[0]

# %%

qs = np.linspace(0.1, 1, 10)
Rs = np.linspace(200, 650, 10)
es = np.logspace(-2, -0.01, 50)
print(es)
# %%

es = np.logspace(-7, 0, 8)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlim(1.24575e9, 1.24725e9)
plt.ylim(1e-7, 1)
plt.yscale("log")
plt.xlabel("Time (yr)")
plt.ylabel("Eccentricity")


for e in es:
    q = 0.5
    R = 500
    e_init = e
    a_init = inv_roche_lobe(R, q)
    [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
        Star, q, a_init, simple_only=True, e_init=e_init
    )
    bin = Bins[0]
    plt.plot(bin.age, bin.e)
plt.savefig("/home/koen/LaTeX-setup/plots/w16-eccentricity-1.pgf", format="pgf")
plt.show()
plt.close()
# %%
import pandas as pd

rows = []

for q in qs:
    print(q)

    for R in Rs:
        print(R)

        for e in es:
            print(e)
            a_init = inv_roche_lobe(R, q)
            [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
                Star, q, a_init, simple_only=True, e_init=e
            )

            bin = Bins[0]

            rows.append(
                {
                    "q": q,
                    "R": R,
                    "e_init": e,
                    "age": bin.age[-1],
                    "a": bin.a[-1],
                    "e": bin.e[-1],
                    "spin1": bin.spin1[-1],
                    "type1": bin.type1[-1],
                    "m1": bin.m1[-1],
                    "m2": bin.m2[-1],
                    "amloss": bin.amloss[-1],
                }
            )

df = pd.DataFrame(rows)
df.to_parquet("scripts/w15-w16/results.parquet")
# %%

print(df)
df.to_csv("scripts/w15-16/results.csv", index=False)
# %%

df = pd.read_csv("scripts/w15-16/results.csv")

# %%
print(df)
# %%
from scipy.interpolate import griddata

qs_unique = np.sort(df["q"].unique())

fig, axes = plt.subplots(
    2, 5, sharex=True, sharey=True, figsize=set_size(full), constrained_layout=True
)


axes = axes.flatten()

for ax, q in zip(axes, qs_unique):

    sub = df[df["q"] == q]

    x = sub["e_init"].values
    y = sub["R"].values
    z = np.log10(np.clip(sub["e"].values, 1e-10, None))

    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    yi = np.linspace(y.min(), y.max(), 200)

    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method="linear")

    levels = [-7, -6, -5, -4, -3, -2, -1, 0]
    levels = np.arange(-7, 0, 0.5)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", extend="both")
    cf.set_rasterized(True)

    ax.set_xscale("log")
    ax.set_title(f"q = {q:.2f}")

axes[7].set_xlabel(r"$\log(\textrm{Initial eccentricity})$")
fig.supylabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.colorbar(
    cf,
    ax=axes.tolist(),
    label=r"$\log_{10}(e_\mathrm{final})$",
    orientation="horizontal",
    aspect=50,
)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w16-eccentricity-2.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
# %%

qs = [0.8]
Rs = np.linspace(200, 650, 50)
es = np.linspace(0.1, 0.99, 50)


rows = []

for q in qs:
    print(q)

    for R in Rs:
        print(R)

        for e in es:
            print(e)
            a_init = inv_roche_lobe(R, q)
            [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
                Star, q, a_init, simple_only=True, e_init=e
            )

            bin = Bins[0]

            rows.append(
                {
                    "q": q,
                    "R": R,
                    "e_init": e,
                    "age": bin.age[-1],
                    "a": bin.a[-1],
                    "e": bin.e[-1],
                    "spin1": bin.spin1[-1],
                    "type1": bin.type1[-1],
                    "m1": bin.m1[-1],
                    "m2": bin.m2[-1],
                    "amloss": bin.amloss[-1],
                }
            )

df = pd.DataFrame(rows)
df.to_csv("scripts/w15-16/results-zoom.csv", index=False)
# %%

df2 = pd.read_csv("scripts/w15-16/results-zoom.csv")
# %%
from scipy.interpolate import griddata

qs_unique = np.sort(df["q"].unique())

fig, axes = plt.subplots(
    1, 1, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)


axes = [axes]

for ax, q in zip(axes, qs_unique):

    sub = df[df["q"] == q]

    x = sub["e_init"].values
    y = sub["R"].values
    z = np.clip(sub["e"].values / sub["e_init"].values, 1e-10, None)

    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    yi = np.linspace(y.min(), y.max(), 200)

    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method="nearest")

    levels = np.arange(-7, 0, 0.5)
    levels = np.arange(0, 1.1, 0.1)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", extend="both")
    cf.set_rasterized(True)

    ax.set_title(f"q = {q:.2f}")

axes[0].set_xlabel(r"Initial eccentricity")
fig.supylabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.colorbar(
    cf,
    ax=axes,
    label=r"$e_\mathrm{final} / e_\mathrm{initial}$",
    orientation="vertical",
)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w16-eccentricity-3.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()

# %%
from scipy.interpolate import griddata

qs_unique = np.sort(df["q"].unique())

fig, axes = plt.subplots(
    1, 1, sharex=True, sharey=True, figsize=set_size(column), constrained_layout=True
)


axes = [axes]

for ax, q in zip(axes, qs_unique):

    sub = df[df["q"] == q]

    x = sub["e_init"].values
    y = sub["R"].values
    z = np.log10(np.clip(sub["e"].values / sub["e_init"].values, 1e-10, None))

    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    yi = np.linspace(y.min(), y.max(), 200)

    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method="nearest")

    levels = np.arange(-3, 0, 0.1)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", extend="both")
    cf.set_rasterized(True)

    ax.set_title(f"q = {q:.2f}")

axes[0].set_xlabel(r"Initial eccentricity")
fig.supylabel("Initial Roche lobe radius ($R_\\odot$)", fontsize=10)
fig.colorbar(
    cf,
    ax=axes,
    label=r"$e_\mathrm{final} / e_\mathrm{initial}$",
    orientation="vertical",
)

plt.savefig(
    "/home/koen/LaTeX-setup/plots/w16-eccentricity-4.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()


# %%

qs = [0.8]
Rs = [550]
es = np.linspace(0.1, 0.99, 50)


rows = []

for q in qs:
    print(q)

    for R in Rs:
        print(R)

        for e in es:
            print(e)
            a_init = inv_roche_lobe(R, q)
            [Star, Options, q_init, a_init, e_init, Bins] = call_evolution(
                Star, q, a_init, simple_only=True, e_init=e
            )

            bin = Bins[0]

            rows.append(
                {
                    "q": q,
                    "R": R,
                    "e_init": e,
                    "age": bin.age,
                    "a": bin.a,
                    "e": bin.e,
                    "spin1": bin.spin1,
                    "type1": bin.type1,
                    "m1": bin.m1,
                    "m2": bin.m2,
                    "amloss": bin.amloss,
                }
            )

df = pd.DataFrame(rows)
df.to_csv("scripts/w15-16/results-row.csv", index=False)
