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

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

from matplotlib.colors import TwoSlopeNorm

from tqdm import tqdm
import pandas as pd

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


def period_to_RL(P, q, m1):

    q_donor = 1 / q
    r_L = (
        0.49
        * q_donor ** (2 / 3)
        / (0.6 * q_donor ** (2 / 3) + np.log1p(q_donor ** (1 / 3)))
    )

    G = 6.67430e-11
    M_total = m1 * (1 + q) * 1.989e30
    R_sun = 695508000

    P = np.asarray(P) * 86400  # days -> seconds

    a = (G * M_total * P**2 / (4 * np.pi**2)) ** (1 / 3)

    return a * r_L / R_sun


def period_to_a(P, q, m1):

    q_donor = 1 / q

    G = 6.67430e-11
    M_total = m1 * (1 + q) * 1.989e30
    R_sun = 695508000

    P = np.asarray(P) * 86400  # days -> seconds

    a = (G * M_total * P**2 / (4 * np.pi**2)) ** (1 / 3)

    return a / R_sun


def get_rlof_properties(star, bin):
    """
    Get stellar properties at the moment of RLOF.

    Returns
    -------
    m_env_rlof : float
        Envelope mass at RLOF, or nan if no RLOF occurred.
    m_star_rlof : float
        Stellar mass at RLOF, or nan if no RLOF occurred.
    rlof_tpagb : bool
        Whether RLOF occurred during the TP-AGB.
    """

    t_rlof = bin.age[-1]

    # Check whether the binary evolved to the end without RLOF
    if np.isclose(t_rlof, star.age[-1]):
        return np.nan, np.nan, False

    # Interpolate stellar properties onto the binary time
    m_env_rlof = np.interp(t_rlof, star.age, star.m_env)
    m_star_rlof = np.interp(t_rlof, star.age, star.mass)

    # Check whether RLOF happened during the TP-AGB
    t_tpagb = star.age[star.ntpagb]
    rlof_tpagb = t_rlof >= t_tpagb

    return m_env_rlof, m_star_rlof, rlof_tpagb


class Result:
    def __init__(self):
        pass


from concurrent.futures import ProcessPoolExecutor

df = AbundanceTables()

ms = np.arange(1, 3.01, 0.1)
qs = np.linspace(0.1, 1.0, 19)
Ps = np.logspace(np.log10(100), np.log10(30000), 100)
epss = [1e-20, 0.1, 0.2, 0.3, 1]

wind_methods = ["Saladino", "BHL", "fast"]


def run_model(pi):

    results = []

    for mi in ms:
        star = get_star(m=mi)

        for qi in qs:

            for method in wind_methods:
                a_init = period_to_a(pi, qi, mi)
                ri = period_to_RL(pi, qi, mi)
                star, Options, q_init, a_init, e_init, Bins = call_evolution(
                    star, qi, a_init, simple_only=True, wind_model=method
                )
                bin = Bins[0]
                m_env_rlof, m_star_rlof, rlof_tpagb = get_rlof_properties(star, bin)

                for eps in epss:

                    ab = Abundances(
                        None,
                        df,
                        mass=mi,
                        sb=bin,
                        sampling=33,
                        mass_transfer_efficiency=eps,
                    )
                    if ab.m_env[-1] > 0.04:
                        m2f = bin.m2[-1] + ab.m_env[-1] * eps
                    else:
                        m2f = bin.m2[-1]

                    results.append(
                        {
                            "p_init": pi,
                            "r_init": ri,
                            "a_init": a_init,
                            "m1i": mi,
                            "m1f": star.mass[-1],
                            "m_env_rlof": m_env_rlof,
                            "m_star_rlof": m_star_rlof,
                            "rlof_tpagb": rlof_tpagb,
                            "qi": qi,
                            "eps": eps,
                            "m2f": m2f,
                            "s": get_s(ab),
                            "hs": get_hs(ab),
                            "ls": get_ls(ab),
                            "ba": get_ba(ab),
                            "wind": method,
                        }
                    )

    with open(
        f"/home/koen/master-internship/scripts/w36/cache/df2_{pi:.1f}.pkl",
        "wb",
    ) as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return results


from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":

    np.random.shuffle(Ps)
    print(Ps)
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_model, pi) for pi in Ps]

        results = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
            )
        ]

    results = [result for r_results in results for result in r_results]

df_results = pd.DataFrame(results)
print(df_results)

with open(
    f"/home/koen/master-internship/scripts/w36/cache/df2.pkl",
    "wb",
) as f:
    pickle.dump(df_results, f, protocol=pickle.HIGHEST_PROTOCOL)
