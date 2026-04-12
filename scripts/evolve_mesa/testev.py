import pickle

from star_model import read_stellar_models
from orbit_evol import *

if __name__ == "__main__":

    # Directory from which to read the MIST stellar evolution tracks
    Star = read_stellar_models()[0]

    # Initial values of the mass ratio, semi-major axis and eccentricity
    # for which the evolution is to be computed
    q_init, a_init, e_init = [], [], []
    print(
        "\nEnter up to 2 sets of initial mass ratio, semi-major axis [Rsun], and",
        "\neccentricity (3 values separated by spaces, entering nothing will skip)",
    )
    for i in range(2):
        line = input(f">> Set {i+1}: ")
        data = line.split()
        if line == "":
            break
        q_init.append(float(data[0]))
        a_init.append(float(data[1]))
        e_init.append(float(data[2]))

    #    q_init = [0.6, 0.6]
    #    a_init = [75, 100]
    #    e_init = [0.0, 0.5]

    # Set modelling options and parameters
    Options = Empty()
    Options.zams_spin_factor = 0.3
    Options.MB_scale_factor = 0
    #    Options.wind_model = "Saladino"  # "fast" / "BHL" / "Saladino"

    tide_model = input("\n>> Tidal model [Preece/BSE]: ")
    if tide_model != "":
        Options.tide_model = tide_model
    else:
        Options.tide_model = "Preece"  # None / "BSE" / "Preece"

    tidal_freq_scaling = input(">> Tidal frequency scaling [BSE/Duguid/Zahn]: ")
    if tidal_freq_scaling != "":
        Options.tidal_freq_scaling = tidal_freq_scaling
    else:
        Options.tidal_freq_scaling = "BSE"  # "BSE" / "Zahn" / "Duguid"

    tide_max_fconv = input(">> Maximum scaling of convective tide: ")
    if tide_max_fconv != "":
        Options.tide_max_fconv = float(tide_max_fconv)
    else:
        Options.tide_max_fconv = 1e6

    tide_scale_factor = input(">> Overall tidal scaling factor: ")
    if tide_scale_factor != "":
        Options.tide_scale_factor = float(tide_scale_factor)
    else:
        Options.tide_scale_factor = 1.0

    print(
        f"Using {Options.tide_scale_factor} times {Options.tide_model} tides with {Options.tidal_freq_scaling} scaling, capped at {Options.tide_max_fconv}"
    )

    filename = input("\n>> Output pickle file (<name>.pkl): ")

    Bins = []
    # Compute and save the orbital evolution for the above system(s)
    for i, (q0, a0, e0) in enumerate(zip(q_init, a_init, e_init)):

        # Compute orbital evolution
        Bin = evolve_orbit(Star, q0, a0, e0, Options, solve_method="LSODA", verbosity=1)

        Bins.append(Bin)

    with open(filename + ".pkl", "wb") as file:
        pickle.dump([Star, Options, q_init, a_init, e_init, Bins], file)
