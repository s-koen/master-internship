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
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid import MesaGrid

# %%

ferguson = MesaGrid(f"{MASTER}/compare-opacity-mass-transfer/ferguson/")
aesopus = MesaGrid(f"{MASTER}/compare-opacity-mass-transfer/aesopus/")
# %%

for i, (q, model) in enumerate(ferguson.get_r1_slice(r1=420)):
    plt.plot(model.age, model.star.R, c=f"c{i}")
    plt.plot(model.age, model.star.rl_1, c=f"c{i}")

for i, (q, model) in enumerate(aesopus.get_r1_slice(r1=420)):
    plt.plot(model.age, model.star.R, c=f"c{i}", alpha=0.5, linewidth=5)
    plt.plot(model.age, model.star.rl_1, c=f"c{i}", alpha=0.5, linewidth=5)

plt.show()
# %%

for i, (q, model) in enumerate(ferguson.get_q_slice(q=0.25)):
    plt.plot(model.age, model.star.R, c=f"C{i}")
plt.plot(ferguson.ref_tpagb.star_age, ferguson.ref_tpagb.R, c=f"C9")

for i, (q, model) in enumerate(aesopus.get_q_slice(q=0.25)):
    plt.plot(model.age, model.star.R, c=f"C{i}", alpha=0.5, linewidth=5)
plt.plot(aesopus.ref_tpagb.star_age, aesopus.ref_tpagb.R, c=f"C9")

plt.show()
# %%

for i, (q, model) in enumerate(ferguson.get_q_slice(q=0.25)):
    plt.plot(model.env_mass, model.star.R, c=f"C{i}")
    plt.plot(model.env_mass, model.star.rl_1, c=f"C{i}")

for i, (q, model) in enumerate(aesopus.get_q_slice(q=0.25)):
    plt.plot(model.env_mass, model.star.R, c=f"C{i}", alpha=0.5, linewidth=5)
    plt.plot(model.env_mass, model.star.rl_1, c=f"C{i}", alpha=0.5, linewidth=5)

plt.show()
# %%
q_choice = 1

for i, (q, model) in enumerate(ferguson.get_q_slice(q=q_choice)):
    plt.plot(model.env_mass, model.star.R / model.star.rl_1, c=f"C{i}")

for i, (q, model) in enumerate(aesopus.get_q_slice(q=q_choice)):
    plt.plot(
        model.env_mass,
        model.star.R / model.star.rl_1,
        c=f"C{i}",
        alpha=0.5,
        linewidth=5,
    )

plt.show()
# %%
