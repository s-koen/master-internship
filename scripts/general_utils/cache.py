import pickle
from pathlib import Path
import sys

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

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"


def get_star(
    proj_dir="/home/koen/master-internship/mesa-models/",
    m=2,
    z="0.00557",
    full_path=None,
):

    if full_path == None:
        directory = (
            Path(proj_dir) / "single-stars" / f"z{z}" / "completed" / f"M{m:.1f}"
        )

    else:
        directory = Path(full_path)

    path = directory / "combined_star.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    star = read_stellar_models(directory)[0]

    with path.open("wb") as f:
        pickle.dump(star, f, protocol=pickle.HIGHEST_PROTOCOL)

    return star
