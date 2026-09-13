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
    print(star)
    plt.plot(
        (ab.time[star.ntpagb :] - ab.time[star.ntpagb])
        / (ab.time[-1] - ab.time[star.ntpagb]),
        ab.c.m_yield[star.ntpagb :],
        c=cmap(norm(m)),
    )
    print(ab.c.envelope[star.ntpagb :])
    print(ab.c.m_yield[star.ntpagb :])

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
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis


for m in np.arange(1.9, 2.101, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    final.append(ab.h.m_yield[-1])
    final.append(ab.he.m_yield[-1])
    z.append(1)
    z.append(2)
    for el in list(ab.df.elements)[7:]:
        e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
        final.append(e)
        data = ab.df.envelope[ab.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))
        # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])

    plt.plot(z, final, label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$", linewidth=1)

m_yield = yields.query(
    "`Initial mass` == 2 and metallicity == 0.0028 and M_mix == 0.006"
)
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"], m_yield["Mass(i)"], label="$M=2\\;M_\\odot,\\;Z=0.0028$", linewidth=1
)

m_yield = yields.query("`Initial mass` == 2.1 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"], m_yield["Mass(i)"], label="$M=2.1\\;M_\\odot,\\;Z=0.007$", linewidth=1
)

m_yield = yields.query("`Initial mass` == 1.9 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"], m_yield["Mass(i)"], label="$M=1.9\\;M_\\odot,\\;Z=0.007$", linewidth=1
)

fig.legend(loc="outside upper center", ncols=3)

labels = ["p", "he"] + list(ab.df.elements)[7:]

plt.xlabel("Z")
plt.xticks(z[::2], labels=labels[::2])

labels = ["p", "he"] + list(ab.df.elements)[7:]
ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylabel("$M_X$ ($M_\\odot$)")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-all.pgf", format="pgf")
plt.show()
plt.close()
# %%

m_yield = yields.query("`Initial mass` == 2 and Z == 0.0028 and M_mix == 0.006")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(m_yield["Mass(i)"])
plt.show()
# %%
plt.plot(m_yield["Mass(i)"])
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
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis

m_yield = yields.query("`Initial mass` == 2.1 and metallicity == 0.007")

for m in np.arange(1.9, 2.101, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    els = []
    final.append(ab.h.m_yield[-1])
    final.append(ab.he.m_yield[-1])
    z.append(1)
    z.append(2)
    els.append("h")
    els.append("he")
    for el in list(ab.df.elements)[7:]:
        if el in ["tc", "pm", "po"]:
            continue
        e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
        final.append(e)
        data = ab.df.envelope[ab.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))
        # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
        els.append(el)

    plt.plot(
        z,
        np.array(final) / np.array(m_yield["Mass(i)"]),
        label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$",
        linewidth=1,
    )

m_yield_new = yields.query(
    "`Initial mass` == 2 and metallicity == 0.0028 and M_mix == 0.006"
)
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.array(m_yield_new["Mass(i)"]) / np.array(m_yield["Mass(i)"]),
    label="$M=2\\;M_\\odot,\\;Z=0.0028$",
    linewidth=1,
)

m_yield_new = yields.query("`Initial mass` == 2.1 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.array(m_yield_new["Mass(i)"]) / np.array(m_yield["Mass(i)"]),
    label="$M=2.1\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)

m_yield_new = yields.query("`Initial mass` == 1.9 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.array(m_yield_new["Mass(i)"]) / np.array(m_yield["Mass(i)"]),
    label="$M=1.9\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)

fig.legend(loc="outside upper center", ncols=3)

labels = els

plt.xlabel("Z")
plt.xticks(z[::2], labels=labels[::2])

labels = ["p", "he"] + list(ab.df.elements)[7:]
ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2][:-1])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylim(0.1, 10)
plt.ylabel("$M_X / M_{X,M=2.1\\;M_\\odot,\\;Z=0.007}$")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-all-diff.pgf", format="pgf")
plt.show()
plt.close()
# %%

print(els)
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


for i, m in enumerate(np.arange(1.9, 2.101, 0.1)):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    axs.axhline(
        star.mass[0] - star.m_core[-1], linewidth=0.75 / 2, color=f"C{i}", alpha=0.5
    )

    final = []
    z = []
    final.append(ab.h.m_yield[-1])
    final.append(ab.he.m_yield[-1])
    z.append(1)
    z.append(2)
    for el in list(ab.df.elements)[7:]:
        e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
        final.append(e)
        data = ab.df.envelope[ab.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))
        # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])

    plt.plot(
        z,
        np.cumsum(final),
        label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$",
        linewidth=1,
    )

m_yield = yields.query(
    "`Initial mass` == 2 and metallicity == 0.0028 and M_mix == 0.006"
)
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.cumsum(m_yield["Mass(i)"]),
    label="$M=2\\;M_\\odot,\\;Z=0.0028$",
    linewidth=1,
)

axs.axhline(2 - 0.659, linewidth=0.75 / 2, color=f"C{i+1}", alpha=0.5)


m_yield = yields.query("`Initial mass` == 2.1 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.cumsum(m_yield["Mass(i)"]),
    label="$M=2.1\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)

axs.axhline(2.1 - 0.659, linewidth=0.75 / 2, color=f"C{i+2}", alpha=0.5)

m_yield = yields.query("`Initial mass` == 1.9 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.cumsum(m_yield["Mass(i)"]),
    label="$M=1.9\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)

axs.axhline(1.9 - 0.657, linewidth=0.75 / 2, color=f"C{i+3}", alpha=0.5)

fig.legend(loc="outside upper center", ncols=3)

labels = ["p", "he"] + list(ab.df.elements)[7:]

plt.xlabel("Z")
plt.xticks(z[::2], labels=labels[::2])

labels = ["p", "he"] + list(ab.df.elements)[7:]
ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylabel("$\sum M_X$ ($M_\\odot$)")
# plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-cum.pgf", format="pgf")
plt.show()
plt.close()
# %%

m_yield.columns
# %%

for m in np.arange(1.9, 2.101, 0.1):
    star = get_star(m=m)
    print(star.mass[0] - star.m_core[-1])

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


for i, m in enumerate(np.arange(1.9, 2.101, 0.1)):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    m_diff = star.mass[0] - star.m_core[-1]

    final = []
    z = []
    final.append(ab.h.m_yield[-1])
    final.append(ab.he.m_yield[-1])
    z.append(1)
    z.append(2)
    for el in list(ab.df.elements)[7:]:
        e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
        final.append(e)
        data = ab.df.envelope[ab.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))
        # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])

    plt.plot(
        z,
        np.cumsum(final) / m_diff,
        label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$",
        linewidth=1,
    )

m_yield = yields.query(
    "`Initial mass` == 2 and metallicity == 0.0028 and M_mix == 0.006"
)

m_diff = 2 - 0.659

# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.cumsum(m_yield["Mass(i)"]) / m_diff,
    label="$M=2\\;M_\\odot,\\;Z=0.0028$",
    linewidth=1,
)


m_yield = yields.query("`Initial mass` == 2.1 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])

m_diff = 2.1 - 0.659
plt.plot(
    m_yield["Z"],
    np.cumsum(m_yield["Mass(i)"]) / m_diff,
    label="$M=2.1\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)


m_yield = yields.query("`Initial mass` == 1.9 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
m_diff = 1.9 - 0.657
plt.plot(
    m_yield["Z"],
    np.cumsum(m_yield["Mass(i)"]) / m_diff,
    label="$M=1.9\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)


fig.legend(loc="outside upper center", ncols=3)

labels = ["p", "he"] + list(ab.df.elements)[7:]

plt.xlabel("Z")
plt.xticks(z[::2], labels=labels[::2])

labels = ["p", "he"] + list(ab.df.elements)[7:]
ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylabel("$\sum M_X / (M_\\textrm{star,i} - M_\\textrm{star,f})$")
plt.ylim(0.980, 1.01)
# plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-cum-div.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)


for i, m in enumerate(np.arange(1.9, 2.101, 0.1)):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)

    final = []
    z = []
    # final.append(ab.h.m_yield[-1])
    # final.append(ab.he.m_yield[-1])
    # z.append(1)
    # z.append(2)
    for el in list(ab.df.elements)[10:]:
        e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
        final.append(e)
        data = ab.df.envelope[ab.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))
        # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])

    plt.plot(
        z,
        np.cumsum(final),
        label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$",
        linewidth=1,
    )

m_yield = yields.query(
    "`Initial mass` == 2 and metallicity == 0.0028 and M_mix == 0.006"
)
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"][5:],
    np.cumsum(m_yield["Mass(i)"][5:]),
    label="$M=2\\;M_\\odot,\\;Z=0.0028$",
    linewidth=1,
)


m_yield = yields.query("`Initial mass` == 2.1 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"][5:],
    np.cumsum(m_yield["Mass(i)"][5:]),
    label="$M=2.1\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)


m_yield = yields.query("`Initial mass` == 1.9 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"][5:],
    np.cumsum(m_yield["Mass(i)"][5:]),
    label="$M=1.9\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)


fig.legend(loc="outside upper center", ncols=3)

labels = list(ab.df.elements)[10:]

plt.xlabel("Z")
plt.xticks(z[::2], labels=labels[::2])

labels = list(ab.df.elements)[10:]
ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylabel("$M_X$ ($M_\\odot$)")
# plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-cum-n-only.pgf", format="pgf")
plt.show()
plt.close()
# %%
m_yield_new = yields.query(
    "`Initial mass` == 1.5 and metallicity == 0.007 and N_ov == 1"
)
m_yield_new
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis

m_yield = yields.query("`Initial mass` == 1.5 and metallicity == 0.007 and N_ov == 1")

for m in np.arange(1.5, 1.501, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    els = []
    final.append(ab.h.m_yield[-1])
    final.append(ab.he.m_yield[-1])
    z.append(1)
    z.append(2)
    els.append("h")
    els.append("he")
    for el in list(ab.df.elements)[7:]:
        if el in ["tc", "pm", "po"]:
            continue
        e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
        final.append(e)
        data = ab.df.envelope[ab.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))
        # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
        els.append(el)

    plt.plot(
        z,
        np.array(final) / np.array(m_yield["Mass(i)"]),
        label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$",
        linewidth=1,
    )

m_yield_new = yields.query(
    "`Initial mass` == 1.5 and metallicity == 0.0028 and M_mix == 0.006"
)
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.array(m_yield_new["Mass(i)"]) / np.array(m_yield["Mass(i)"]),
    label="$M=1.5\\;M_\\odot,\\;Z=0.0028$",
    linewidth=1,
)

m_yield_new = yields.query(
    "`Initial mass` == 1.5 and metallicity == 0.007 and N_ov == 1"
)
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.array(m_yield_new["Mass(i)"]) / np.array(m_yield["Mass(i)"]),
    label="$M=1.5\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)

fig.legend(loc="outside upper center", ncols=3)

labels = els

plt.xlabel("Z")
plt.xticks(z[::2], labels=labels[::2])

labels = ["p", "he"] + list(ab.df.elements)[7:]
ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2][:-1])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylim(0.1, 10)
plt.ylabel("$M_X / M_{X,M=1.5\\;M_\\odot,\\;Z=0.007}$")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-all-diff-m1.5.pgf", format="pgf")
plt.show()
plt.close()
# %%
m_yield_new = yields.query(
    "`Initial mass` == 2.5 and metallicity == 0.0028 and M_mix == 0.004"
)
print(np.unique(yields.M_mix))
m_yield_new

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis

m_yield = yields.query("`Initial mass` == 2.5 and metallicity == 0.007")

for m in np.arange(2.5, 2.501, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    els = []
    final.append(ab.h.m_yield[-1])
    final.append(ab.he.m_yield[-1])
    z.append(1)
    z.append(2)
    els.append("h")
    els.append("he")
    for el in list(ab.df.elements)[7:]:
        if el in ["tc", "pm", "po"]:
            continue
        e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
        final.append(e)
        data = ab.df.envelope[ab.df.envelope["element"] == el]
        z.append(int(np.array(data["elemental_mass"])[0]))
        # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
        els.append(el)

    plt.plot(
        z,
        np.array(final) / np.array(m_yield["Mass(i)"]),
        label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$",
        linewidth=1,
    )


m_yield_new = yields.query(
    "`Initial mass` == 2.5 and metallicity == 0.0028 and M_mix == 0.002"
)

# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.array(m_yield_new["Mass(i)"]) / np.array(m_yield["Mass(i)"]),
    label="$M=2.5\\;M_\\odot,\\;Z=0.0028,\\;M_\\textrm{mix} = 0.002$",
    linewidth=1,
)


m_yield_new = yields.query(
    "`Initial mass` == 2.5 and metallicity == 0.0028 and M_mix == 0.004"
)

# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.array(m_yield_new["Mass(i)"]) / np.array(m_yield["Mass(i)"]),
    label="$M=2.5\\;M_\\odot,\\;Z=0.0028,\\;M_\\textrm{mix} = 0.004$",
    linewidth=1,
)

m_yield_new = yields.query("`Initial mass` == 2.5 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.array(m_yield_new["Mass(i)"]) / np.array(m_yield["Mass(i)"]),
    label="$M=2.5\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)

fig.legend(loc="outside upper center", ncols=2)

labels = els

plt.xlabel("Z")
plt.xticks(z[::2], labels=labels[::2])

labels = ["p", "he"] + list(ab.df.elements)[7:]
ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2][:-1])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylim(0.1, 10)
plt.ylabel("$M_X / M_{X,M=2.5\\;M_\\odot,\\;Z=0.007}$")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w27-yield-all-diff-m2.5.pgf", format="pgf")
plt.show()
plt.close()
# %%
