import mesa_reader as mr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
sys.path.insert(1, "/home/koen/master-internship/")

from scripts.general_utils.mesa_grid import MesaGrid
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")
# %%

grid = MesaGrid("/home/koen/master-internship/mesa-models/binary-tpagb-grid-2/runs/")

print(grid.m1_vals)
print(grid.q_vals)

# %%

count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [300]:
        continue
    print(q)
    hist = model["tpagb_history"]
    plt.plot(hist.star_mass - hist.he_core_mass, hist.R, c=f"C{count}")
    plt.plot(
        hist.star_mass - hist.he_core_mass,
        hist.rl_1,
        c=f"C{count}",
        linewidth=5,
        alpha=0.5,
    )
    count += 1

plt.gca().invert_xaxis()
plt.show()


# %%
def rol(history):
    q = history.star_2_mass / history.star_1_mass
    return history.rl_1 * (1 + (0.441 * q ** (-0.325)) / (1 + 0.412 * q ** (-0.8)))


# %%


count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [300]:
        continue
    print(q)
    hist = model["tpagb_history"]
    plt.plot(hist.star_mass - hist.he_core_mass, hist.R / hist.rl_1, c=f"C{count}")
    # plt.plot(
    #     hist.star_mass - hist.he_core_mass,
    #     hist.R / rol(hist),
    #     c=f"C{count}",
    #     linewidth=5,
    #     alpha=0.3,
    # )
    count += 1

plt.gca().invert_xaxis()
plt.ylim(0.7, 3)
plt.show()
# %%


count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [225]:
        continue
    print(q)
    hist = model["tpagb_history"]
    plt.plot(hist.star_age, hist.R, c=f"C{count}")
    plt.plot(
        hist.star_age,
        hist.rl_1,
        c=f"C{count}",
        linewidth=5,
        alpha=0.5,
    )
    count += 1

plt.show()


# %%


count = 0
for m1, q, model in grid.iter_models():
    if m1 not in [225]:
        continue
    print(q)
    hist = model["tpagb_history"]
    plt.plot(hist.log_Teff, hist.log_L, c=f"C{count}")
    count += 1

plt.gca().invert_xaxis()
plt.show()
# %%
