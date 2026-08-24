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
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

grid = MesaGrid(f"{MASTER}grid-masses-2-2026-08-16")

# %%

rng = np.random.default_rng(seed=9)
models = rng.permutation(grid.models)
for model in models:

    if model.period_days[-1] < 50:
        print(model.params)
        break

plt.plot(model.age, model.R)
plt.plot(model.age, model.rl_1)
plt.show()


# %%

fix_model = mr.MesaData(
    "/home/koen/master-internship/mesa-models/grid-masses-2026-08-14-clean/R800.00_q0.400_eps0.250_delta0.200_M3.0/LOGS/history.data"
)
# %%

plt.plot(fix_model.star_age, fix_model.R)
plt.plot(fix_model.star_age, fix_model.rl_1)
plt.plot(model.star_age, model.R)
plt.show()

# %%
