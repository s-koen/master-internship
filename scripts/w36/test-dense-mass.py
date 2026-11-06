import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
from matplotlib.ticker import ScalarFormatter
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.plot_helpers import *
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)
from scripts.general_utils.accretor import *
from scripts.general_utils.asplund import Element, Asplund

plt.cplot = cplot
plt.cplot = annotate_pcolormesh
sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary

# %%


for m in np.arange(1.90, 2.11, 0.01):
    if np.isclose(m, 2.03):
        continue
    star = get_star(m=m)

    plt.plot(
        star.age[star.ntpagb :] - star.age[star.ntpagb], 10 ** star.log_R[star.ntpagb :]
    )
    plt.text(star.age[-1] - star.age[star.ntpagb], 10 ** star.log_R[-1], f"{m:.2f}")

plt.show()
# %%

tp_count = []
ms = []

for m in np.arange(1, 3.01, 0.01):
    if np.isclose(m, 2.03):
        continue
    try:
        star = get_star(m=m)
    except:
        continue

    ms.append(m)
    tp_count.append(star.TP_count[-1])

plt.plot(ms, tp_count)
plt.show()
# %%

tp_count = []
ms = []

for m in np.arange(1, 3.01, 0.01):
    if np.isclose(m, 2.03):
        continue
    try:
        star = get_star(m=m)
    except:
        continue

    ms.append(m)
    tp_count.append(star.m_DUP_time[0][-1])

plt.plot(ms, tp_count)
plt.show()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full, height=0.5), constrained_layout=True
)

ms = []

for m in np.arange(1, 3.01, 0.01):
    try:
        star = get_star(m=m)
    except:
        continue

    ms.append(m)
    max_TP = int(np.max(star.TP_count))
    print(max_TP)
    mm = []
    m_envs = []
    for p in range(1, max_TP + 1):
        mm.append(m)
        ind = np.argwhere(star.TP_count == p)[-1][0]
        m_envs.append(star.m_env[ind])
    plt.plot(mm, m_envs, c="k", linewidth=1)
    plt.scatter(
        mm[:-1],
        m_envs[:-1],
        c="k",
        marker=".",
        s=[np.max([30 - 3 * x, 1]) for x in range(max_TP + 1, 1, -1)][:-1],
        zorder=10,
    )
    plt.scatter(
        mm[:-1],
        m_envs[:-1],
        c="w",
        marker=".",
        s=2
        * np.array([np.max([30 - 3 * x, 1]) for x in range(max_TP + 1, 1, -1)])[:-1],
        zorder=9,
    )


# plt.yscale("log")
# plt.xscale("log")

plt.xticks(np.arange(1, 3.01, 0.2))

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{TPAGB}$ ($M_\\odot$)")
plt.ylabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-mass-at-TP.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.9), constrained_layout=True
)

ms = []

minus_one = []
minus_two = []
minus_three = []
zero = []

for m in np.arange(1, 3.01, 0.01):
    try:
        star = get_star(m=m)
    except:
        continue

    ms.append(m)
    max_TP = int(np.max(star.TP_count))
    print(max_TP)
    mm = []
    m_envs = []
    for p in range(1, max_TP + 1):
        mm.append(m)
        ind = np.argwhere(star.TP_count == p)[-1][0]
        m_envs.append(star.m_env[ind])
    minus_one.append(m_envs[-2])
    minus_two.append(m_envs[-3])
    minus_three.append(m_envs[-4])
    zero.append(m_envs[0])


plt.scatter(ms, minus_one)
plt.scatter(ms, minus_two)
plt.scatter(ms, minus_three)
plt.scatter(ms, zero)
# plt.yscale("log")
# plt.xscale("log")

plt.xticks(ms[::2])

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$M_\\textrm{TPAGB}$ ($M_\\odot$)")
plt.ylabel("$m_\\textrm{env}$ ($M_\\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w36-mass-at-TP.pgf", format="pgf")
plt.show()
plt.close()
# %%
from cmap import Colormap

cm = Colormap("bids:fake_parula_r").to_mpl()  # case insensitive
fig, axs = plt.subplots(
    5, 4, sharey=True, figsize=set_size(full, height=1.75), constrained_layout=True
)

axs = axs.flatten()


i = 0

cmin = 1e99
cmax = -1e99
for m in np.arange(1.90, 2.10, 0.01):
    try:
        star = get_star(m=m)
    except:
        continue

    if np.nanmin(star.m_env[star.ntpagb :]) < cmin:
        cmin = np.nanmin(star.m_env[star.ntpagb :])

    if np.nanmax(star.m_env[star.ntpagb :]) > cmax:
        cmax = np.nanmax(star.m_env[star.ntpagb :])


for m in np.arange(1.90, 2.10, 0.01):
    try:
        star = get_star(m=m)
    except:
        continue

    axs[i].set_xlim(
        star.age[-1]
        - star.age[star.ntpagb]
        - (star.age[-1] - star.age[star.ntpagb]) / 5,
        1.01 * (star.age[-1] - star.age[star.ntpagb]),
    )

    ls = cplot(
        star.age[star.ntpagb :: 30] - star.age[star.ntpagb],
        10 ** star.log_R[star.ntpagb :: 30],
        star.m_env[star.ntpagb :: 30],
        cmap=cm,
        ax=axs[i],
        vmin=cmin,
        vmax=cmax,
    )

    for mm in np.arange(1.90, 2.10, 0.01):
        try:
            star = get_star(m=mm)
        except:
            continue
        if m == mm:
            continue

        alpha = np.max([1 - (np.abs(m - mm) * 5), 0.1])
        print(alpha**4)

        axs[i].plot(
            star.age[star.ntpagb :] - star.age[star.ntpagb],
            10 ** star.log_R[star.ntpagb :],
            c="C9",
            linewidth=0.75,
            alpha=alpha**4,
            zorder=-10,
            rasterized=True,
        )

    axs[i].text(
        0.05,
        0.95,
        f"${m:.2f}\\;M_\\odot$",
        transform=axs[i].transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )
    i += 1

fig.colorbar(
    ls,
    ax=axs[1:3],
    label="$M_\\textrm{env}$ ($M_\\odot$)",
    orientation="horizontal",
    location="top",
)

for i, ax in enumerate(axs):
    ax.spines[["right", "top"]].set_visible(False)
    if i % 4 == 0:
        ax.set_ylabel("$R$ ($R_\\odot$)")
    if i > 15:
        ax.set_xlabel("Star age (yr)")


plt.savefig(
    "/home/koen/LaTeX-setup/plots/w36-diff-radial-evolution.pgf", format="pgf", dpi=500
)
plt.show()
plt.close()

# %%
