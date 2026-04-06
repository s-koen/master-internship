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

# %%
no_tides = mr.MesaData(
    "/home/koen/master-internship/mesa-models/binary-tpagb-grid-4/runs/R506.00_q0.800/LOGS/TPAGB/history.data"
)
# %%
wind_fast = mr.MesaData(f"{MASTER}/wind/2/LOGS/TPAGB/history.data")
# %%
wind_sal = mr.MesaData(f"{MASTER}/wind/4/LOGS/TPAGB/history.data")
# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.75), constrained_layout=True
)

# plt.plot(hist1.age, hist1.R, c="C9")

index = np.where(no_tides.R / no_tides.rl_1 > 1)[0][0]
plt.plot(
    no_tides.age[:index],
    no_tides.rl_1[:index],
    label="No tides + fast wind",
    zorder=1,
    c="C2",
)
plt.scatter(
    no_tides.age[index],
    no_tides.rl_1[index],
    zorder=1,
    c="C2",
)
plt.plot(
    no_tides.age[:index], no_tides.R[:index], c="C9", alpha=0.3, linewidth=3, zorder=-10
)

index = np.where(wind_fast.R / wind_fast.rl_1 > 1)[0][0]
plt.plot(
    wind_fast.age[:index],
    wind_fast.rl_1[:index],
    label="Tides + fast wind",
    c="C3",
    zorder=10,
)
plt.scatter(wind_fast.age[index], wind_fast.rl_1[index], c="C3", zorder=10)
plt.plot(
    wind_fast.age[:index],
    wind_fast.R[:index],
    c="C9",
    alpha=0.3,
    linewidth=3,
    zorder=-10,
)

index = np.where(wind_sal.R / wind_sal.rl_1 > 1)[0][0]
plt.plot(
    wind_sal.age[:index],
    wind_sal.rl_1[:index],
    label="Tides + Saladino",
    linestyle="-",
    c="C0",
    zorder=300,
)
plt.scatter(wind_sal.age[index], wind_sal.rl_1[index], c="C0", zorder=300)

plt.plot(
    wind_sal.age[:index], wind_sal.R[:index], c="C9", alpha=0.3, linewidth=3, zorder=-10
)

fig.legend(loc="outside upper center", ncols=3)
plt.ylim(325)
plt.xlim(0, 375000)
plt.xlabel(r"Time (yr)")
plt.ylabel(r"Roche lobe radius ($R_\odot$)")
plt.xticks(
    [0, 125000, 250000, 325000, 375000],
    ["0", "$125\,000$", "$250\,000$", "wind", "$375\,000$"],
)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w10-stellar-evolution-meeting.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
