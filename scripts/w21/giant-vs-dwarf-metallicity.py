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

import mesa_reader as mr
import pandas as pd
from scripts.general_utils.mesa_grid import MesaGrid

# %%

data = pd.read_csv("presentation-1/Ba_star_orbits.csv")


# %%
for c in data["class"]:
    print(c)

# %%
data.columns
# %%
dwarf = data[data["class"].isin(["dBa", "?dBa"])]
mild = data[
    data["class"].isin(
        ["Ba 0", "Ba 0.5", "Ba 1", "Ba 2", "Ba mild", "sgCH", "sgCH / Ba 1"]
    )
]

strong = data[data["class"].isin(["Ba 3", "Ba 4", "Ba 5", "Ba strong", "Ba 5 / eS"])]
# %%
plt.plot(dwarf["[Fe/H]"], dwarf["[Fe/H]"] * 0)
plt.plot(mild["[Fe/H]"], mild["[Fe/H]"] * 0 + 1)
plt.plot(strong["[Fe/H]"], strong["[Fe/H]"] * 0 + 1)

plt.show()
# %%
