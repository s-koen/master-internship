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
plt.cplot = annotate_pcolormesh
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary

sys.path.insert(1, "/home/koen/master-internship/")

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

# %%

with open(
    f"/home/koen/master-internship/scripts/w31/cache/df.pkl",
    "rb",
) as f:
    df_results = pickle.load(f)
df = AbundanceTables()


def get_ls(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 3


def get_hs(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in [
            "ba",
            "la",
            "ce",
            "nd",
        ]:
            fe += ab.iron_abundance[i]

    return fe / 4


def get_s(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += ab.iron_abundance[i]

    return fe / 7


def get_ba(ab):
    fe = 0
    for i, name in enumerate(ab.elements_name):

        if name in ["ba"]:
            fe += ab.iron_abundance[i]

    return fe


# %%
def evaluate_R(ri, mi, eps, qi=0.75):
    star = get_star(m=mi)

    a_init = inv_roche_lobe(ri, qi)
    star, Options, q_init, a_init, e_init, Bins = call_evolution(
        star, qi, a_init, simple_only=True
    )
    bin = Bins[0]

    ab = Abundances(
        None,
        df,
        mass=mi,
        sb=bin,
        sampling=33,
        mass_transfer_efficiency=eps,
    )

    s = get_s(ab)

    if ab.m_env[-1] > 0.04:
        m2f = bin.m2[-1] + ab.m_env[-1] * eps
    else:
        m2f = bin.m2[-1]

    return {
        "r_init": ri,
        "a_init": a_init,
        "m1i": mi,
        "m1f": star.mass[-1],
        "qi": qi,
        "eps": eps,
        "m2f": m2f,
        "s": s,
        "hs": get_hs(ab),
        "ls": get_ls(ab),
        "ba": get_ba(ab),
    }


def find_min_R(mi, eps, R_low, R_high, tol=1.0, threshold=0.25):
    """
    Find the lowest R for which s > threshold.

    Assumes s is monotonically increasing with R.
    R_low must fail (s <= threshold).
    R_high must pass (s > threshold).
    """
    low_result = evaluate_R(R_low, mi, eps)
    high_result = evaluate_R(R_high, mi, eps)

    if low_result["s"] > threshold:
        raise ValueError(f"R_low={R_low} already passes: s={low_result['s']}")

    if high_result["s"] <= threshold:
        raise ValueError(f"R_high={R_high} does not pass: s={high_result['s']}")

    while R_high - R_low > tol:
        R_mid = (R_low + R_high) / 2
        mid_result = evaluate_R(R_mid, mi, eps)

        if mid_result["s"] > threshold:
            R_high = R_mid
            high_result = mid_result
        else:
            R_low = R_mid
            low_result = mid_result
        print(R_low, R_high)

    return high_result


# %%
results = []

ms = np.arange(1.2, 3.1, 0.1)
for mi in ms:
    for eps in [1]:
        result = find_min_R(
            mi=mi,
            eps=eps,
            R_low=50,
            R_high=800,
            tol=1.0,
        )
        results.append(result)

results = pd.DataFrame(results)
# %%


plt.plot(results["m1i"], results["r_init"])
plt.show()
# %%
results_0 = []

ms = np.arange(1.2, 3.1, 0.1)
for mi in ms:
    for eps in [0]:
        result = find_min_R(
            mi=mi,
            eps=eps,
            R_low=50,
            R_high=800,
            tol=1.0,
        )
        results_0.append(result)

results_0 = pd.DataFrame(results_0)
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(eps1["m1i"], eps1["r_init"], label="$\\varepsilon=1$", c="C0")
plt.scatter(eps1["m1i"], eps1["r_init"], c="C0", zorder=100, marker=".", s=50)
plt.scatter(eps1["m1i"], eps1["r_init"], c="w", zorder=99, marker=".", s=100)
plt.plot(eps0["m1i"], eps0["r_init"], label="$\\varepsilon=0$", c="C1")
plt.scatter(eps0["m1i"], eps0["r_init"], c="C1", zorder=100, marker=".", s=50)
plt.scatter(eps0["m1i"], eps0["r_init"], c="w", zorder=99, s=100, marker=".")

axs.annotate(
    "$\\varepsilon_\\textrm{RLOF} = 1$",
    xy=(2.1, 440),
    xycoords="data",
    xytext=(0, -25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
    ha="center",
)

axs.annotate(
    "$\\varepsilon_\\textrm{RLOF} = 0$",
    xy=(2.1, 310),
    xycoords="data",
    xytext=(0, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
    ha="center",
)

plt.title("$q = 0.75$", fontsize=10)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{TPAGB}$ ($M_\\odot$)")
plt.ylabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w34-min-R_RL-for-barium-star.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

with open(
    f"/home/koen/master-internship/scripts/w34/cache/eps0.pkl",
    "wb",
) as f:
    pickle.dump(results_0, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(
    f"/home/koen/master-internship/scripts/w34/cache/eps1.pkl",
    "wb",
) as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%

with open(
    f"/home/koen/master-internship/scripts/w34/cache/eps1.pkl",
    "rb",
) as f:
    eps1 = pickle.load(f)

with open(
    f"/home/koen/master-internship/scripts/w34/cache/eps0.pkl",
    "rb",
) as f:
    eps0 = pickle.load(f)
# %%
