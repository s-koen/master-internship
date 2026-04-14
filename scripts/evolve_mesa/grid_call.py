from star_model import read_stellar_models
from orbit_evol import *


def call_evolution(Star, q, a_init, simple_only=False):
    # Initial values of the mass ratio, semi-major axis and eccentricity
    # for which the evolution is to be computed

    q_init = [q]
    a_init = [a_init]
    e_init = [0]

    # Set modelling options and parameters
    Options = Empty()
    Options.zams_spin_factor = 0.3
    Options.MB_scale_factor = 0

    Options.tide_model = "Preece"  # None / "BSE" / "Preece"
    Options.tidal_freq_scaling = "BSE"  # "BSE" / "Zahn" / "Duguid"
    Options.tide_max_fconv = 1e6
    Options.tide_scale_factor = 1.0
    Options.simple_only = simple_only

    Bins = []
    # Compute and save the orbital evolution for the above system(s)
    for i, (q0, a0, e0) in enumerate(zip(q_init, a_init, e_init)):

        # Compute orbital evolution
        Bin = evolve_orbit(Star, q0, a0, e0, Options, solve_method="LSODA", verbosity=1)

        Bins.append(Bin)

    return [Star, Options, q_init, a_init, e_init, Bins]
