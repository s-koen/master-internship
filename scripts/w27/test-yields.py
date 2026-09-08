import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pickle

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

sys.path.insert(1, "/home/koen/master-internship/")
from scripts.general_utils.cplot import cplot
from scripts.general_utils.cache import get_star
from scripts.general_utils.m_dup import compute_m_DUP, AbundanceTables, Abundances

plt.cplot = cplot

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid

# %%

with open("data/yields_pd_df.pkl", "rb") as f:
    yields = pickle.load(f)

# %%

df = AbundanceTables()

ab = Abundances(None, df, method="tp offset", mass=2.5)
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m in np.arange(1.0, 3.1, 0.2):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    plt.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.c.m_yield[star.ntpagb :],
        c=cmap(norm(m)),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(f"$M_\\textrm{{TPAGB}} = {m:.1f}\;M_\\odot$")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Normalized TPAGB age")
plt.ylabel("$M_\\textrm{C}$ ($M_\\odot$)")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-mesa.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m in np.arange(1.0, 3.1, 0.2):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    plt.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.ba.m_yield[star.ntpagb :],
        c=cmap(norm(m)),
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(f"$M_\\textrm{{TPAGB}} = {m:.1f}\;M_\\odot$")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Normalized TPAGB age")
plt.ylabel("$M_\\textrm{Ba}$ ($M_\\odot$)")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-mesa-ba.pgf", format="pgf")
plt.show()
plt.close()
# %%
for el in ab.df.elements:
    print(ab.df.elements[el])
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis
# color = cmap(norm(x))


for m in np.arange(1.9, 2.2, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    final.append(ab.p.m_yield[-1])
    final.append(ab.he.m_yield[-1])
    z.append(1)
    z.append(2)
    for el in list(ab.df.elements)[7:]:
        if el == "p":
            continue
        e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
        final.append(e)
        data = ab.df.envelope[ab.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))
        # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])

    plt.plot(z, final, f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$")

m_yield = yields.query(
    "`Initial mass` == 2 and metallicity == 0.0028 and M_mix == 0.006"
)
# plt.plot(m_yield["El"], m_yield["X(i)"])
plt.plot(m_yield["Z"], m_yield["X(i)"], label="$M=2\\;M_\\odot,\\;Z=0.0028$")

m_yield = yields.query("`Initial mass` == 2.1 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["X(i)"])
plt.plot(m_yield["Z"], m_yield["X(i)"], label="$M=2.1\\;M_\\odot,\\;Z=0.007$")

m_yield = yields.query("`Initial mass` == 1.9 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["X(i)"])
plt.plot(m_yield["Z"], m_yield["X(i)"], label="$M=1.9\\;M_\\odot,\\;Z=0.007$")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label(f"$M_\\textrm{{TPAGB}} = {m:.1f}\;M_\\odot$")
fig.legend(loc="outside upper center", ncols=3)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Normalized TPAGB age")
plt.ylabel("$M_\\textrm{Ba}$ ($M_\\odot$)")
plt.yscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-mesa-ba.pgf", format="pgf")
plt.show()
plt.close()
# %%

m_yield = yields.query("`Initial mass` == 2 and Z == 0.0028 and M_mix == 0.006")
# plt.plot(m_yield["El"], m_yield["X(i)"])
plt.plot(m_yield["X(i)"])
plt.show()
# %%
plt.plot(m_yield["X(i)"])
plt.show()
# %%

plt.plot(m_yield["El"], m_yield["M_mix"])
plt.show()
# %%
for el in m_yield["El"]:
    print(el)
# %%

ab.df.envelope.columns
# %%

m_yield
# %%

print(ab.df.envelope.columns)
f = ab.df.envelope[ab.df.envelope["element"] == "c"]
f
# %%

np.unique(yields["metallicity"])
# %%
