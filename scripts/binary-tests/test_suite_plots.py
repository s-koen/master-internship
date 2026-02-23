import mesa_reader as mr

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

# %%

binary_history = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-test/binary_history.data"
)

star_history = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-test/LOGS1/history.data"
)
# %%

fig, axs = plt.subplots(1, 2, figsize=set_size(column), constrained_layout=True)

axs[0].plot(binary_history.age, binary_history.lg_mstar_dot_1)
axs[0].set_ylim(-10, -6)
axs[1].plot(star_history.star_age, star_history.star_mass)
axs[0].set_xlabel("System age (yr)")
axs[0].set_ylabel(r"$\log(\dot{M} / (M_\odot \textrm{yr}^{-1}))$")
axs[1].set_xlabel("Star age (yr)")
axs[1].set_ylabel(r"Donor mass ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/binary-test-plot-1.pgf", format="pgf")
plt.show()
plt.close()

# %%
