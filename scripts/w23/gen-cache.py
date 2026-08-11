import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from tqdm import tqdm
import sys
import pickle
from pathlib import Path
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
sys.path.insert(1, "/home/koen/master-internship/scripts/evolve_mesa/")
sys.path.insert(1, "/home/koen/master-internship/")

from scripts.evolve_mesa.constants import *
from scripts.evolve_mesa.bin_input import *
from scripts.evolve_mesa.read_mist_models import *
from scripts.evolve_mesa.mrenv import *
from scripts.evolve_mesa.orbit_evol import *
from scripts.evolve_mesa.rgbf import *
from scripts.evolve_mesa.star_model import *
from scripts.evolve_mesa.grid_call import *

proj_dir = "/home/koen/master-internship"
single_star_dir = f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M1.5"

cache_dir = Path(f"{proj_dir}/scripts/w23/cache")
cache_dir.mkdir(exist_ok=True)


def compute_epsilon(barium_mass, tpagb_mass, core_mass, q):
    """
    computes the mass transfer efficiency assuming no wind mass transfer.
    """
    delta_donated = tpagb_mass - core_mass
    delta_accreted = barium_mass - tpagb_mass * q

    return delta_accreted / delta_donated


def compute_final_mass(epsilon, tpagb_mass, core_mass, q):
    """
    computes the final mass of the barium star for a given epsilon
    """
    delta_donated = tpagb_mass - core_mass
    delta_accreted = epsilon * delta_donated

    return delta_accreted + tpagb_mass * q


for m_i in np.arange(1.0, 2.55, 0.1):

    single_star_dir = (
        f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
    )
    try:
        with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
            Star = pickle.load(f)
        print("read back combined_star.pkl")
    except:

        Star = read_stellar_models(single_star_dir)[0]
        with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
            pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

barium_mass_min = 0.73
barium_mass_max = 1.5
eps_max = 0.5

masses = [3.0]
print(masses)

qs = np.arange(0.05, 1.05, 0.05)
rs = np.arange(100, 1260, 10)
possible_matrix = np.full((len(rs), len(qs)), np.nan)
eps_min_matrix = np.full((len(rs), len(qs)), np.nan)
eps_max_matrix = np.full((len(rs), len(qs)), np.nan)
m_DUP_matrix = np.full((len(rs), len(qs)), np.nan)
CO_matrix = np.full((len(rs), len(qs)), np.nan)
min_mass_matrix = np.full((len(rs), len(qs)), np.nan)
max_mass_matrix = np.full((len(rs), len(qs)), np.nan)
wind_accretion_matrix = np.full((len(rs), len(qs)), np.nan)

for m_i in masses:

    for i, R in enumerate(tqdm(rs)):

        single_star_dir = (
            f"{proj_dir}/mesa-models/single-stars/z0.00557/completed/M{m_i:.1f}"
        )
        try:
            with open(f"{single_star_dir}/combined_star.pkl", "rb") as f:
                Star = pickle.load(f)
        except:

            Star = read_stellar_models(single_star_dir)[0]
            with open(f"{single_star_dir}/combined_star.pkl", "wb") as f:
                pickle.dump(Star, f, protocol=pickle.HIGHEST_PROTOCOL)

        for j, q in enumerate(qs):

            a_init = inv_roche_lobe(R, q)

            cache_file = cache_dir / f"m{m_i:.1f}_R{R:.2f}_q{q:.3f}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)

                if len(result) == 6:
                    # old cache
                    Star, Options, q_init, a_init, e_init, Bins = result
                elif len(result) == 5:
                    # new cache (Star omitted)
                    Options, q_init, a_init, e_init, Bins = result
                else:
                    raise ValueError(
                        f"Unexpected cache format ({len(result)} elements)"
                    )
            else:
                Star, Options, q_init, a_init, e_init, Bins = call_evolution(
                    Star, q, a_init, simple_only=True
                )

                with open(cache_file, "wb") as f:
                    pickle.dump(
                        (Options, q_init, a_init, e_init, Bins),
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            R_star = 10**Star.log_R
            core_mass = Star.m_core
            ages_star = Star.age
            bin = Bins[0]

            q_evolve = bin.m2 / bin.m1
            RL = roche_lobe(1 / q_evolve) * bin.a

            q_begin = q_evolve[-1]
            tpagb_mass = bin.m1[-1]

            # interpolate stellar radius onto binary ages
            core_mass_interp = np.interp(bin.age, ages_star, core_mass)
            try:
                TP_count = int(
                    np.interp(bin.age, Star.age[Star.ntpagb :], Star.TP_count)[-1]
                )
            except ValueError:
                TP_count = int(
                    np.interp(
                        bin.age, Star.age[Star.ntpagb :], Star.TP_count[Star.ntpagb :]
                    )[-1]
                )
            CO_ratio = np.interp(
                bin.age, Star.age, Star.envelope_c12 / Star.envelope_o16 * 16 / 12
            )[-1]

            eps_min_mass = compute_epsilon(
                barium_mass_min, tpagb_mass, core_mass_interp[-1], q_begin
            )
            eps_max_mass = compute_epsilon(
                barium_mass_max, tpagb_mass, core_mass_interp[-1], q_begin
            )

            min_mass = compute_final_mass(0, tpagb_mass, core_mass_interp[-1], q_begin)
            max_mass = compute_final_mass(
                0.5, tpagb_mass, core_mass_interp[-1], q_begin
            )

            if eps_min_mass < 0 and eps_max_mass < 0:
                possible = False
            elif eps_min_mass > eps_max and eps_max_mass > eps_max:
                possible = False
            elif bin.age[-1] - ages_star[-1] == 0:
                possible = 0.5
            else:
                possible = True

            possible_matrix[i, j] = possible
            min_mass_matrix[i, j] = min_mass
            max_mass_matrix[i, j] = max_mass
            eps_min_matrix[i, j] = eps_min_mass
            eps_max_matrix[i, j] = eps_max_mass
            m_DUP_matrix[i, j] = np.sum(Star.m_DUP[:TP_count])
            wind_accretion_matrix[i, j] = bin.m2[-1] - bin.m2[0]
            CO_matrix[i, j] = CO_ratio

    with open(f"scripts/w23/const_mass_possible_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(possible_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_eps_min_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(eps_min_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_eps_max_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(eps_max_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_min_mass_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(min_mass_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_max_mass_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(max_mass_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_co_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(CO_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_dup_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(m_DUP_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"scripts/w23/const_mass_wind_accretion_{m_i:.1f}_small.pkl", "wb") as f:
        pickle.dump(wind_accretion_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
