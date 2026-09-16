import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import matplotlib.transforms as mtransforms
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
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)

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
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

norm = plt.Normalize(1, 3)
cmap = plt.cm.viridis


for m in np.arange(2.5, 2.501, 0.1):
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
    "`Initial mass` == 2.5 and metallicity == 0.0028 and M_mix == 0.002"
)
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    m_yield["Mass(i)"],
    label="$M=2.5\\;M_\\odot,\\;Z=0.0028$",
    linewidth=1,
)

m_yield = yields.query("`Initial mass` == 2.5 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"], m_yield["Mass(i)"], label="$M=2.5\\;M_\\odot,\\;Z=0.007$", linewidth=1
)

fig.legend(loc="outside upper center", ncols=3)

names = ["p", "he"] + list(ab.df.elements)[7:]
labels = []
for name in names:
    labels.append(name.capitalize())

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)


for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylabel("$M_X$ ($M_\\odot$)")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-yield-all.pgf", format="pgf")
plt.show()
plt.close()  # %%
np.unique(yields["metallicity"])

# %%
m_yield_new = yields.query(
    "`Initial mass` == 2.5 and metallicity == 0.0028 and M_mix == 0.004"
)
print(np.unique(yields.M_mix))
m_yield_new

# %%
import matplotlib.transforms as mtransforms

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
labels = []
for el in els:
    labels.append(el.capitalize())

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylim(0.1, 10)
plt.ylabel("$M_X / M_{X,M=2.5\\;M_\\odot,\\;Z=0.007}$")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-yield-all-diff-m2.5.pgf", format="pgf")
plt.show()
plt.close()
# %%

star = get_star(m=2)
print(star)
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in np.arange(2.5, 2.501, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    els = []
    plt.plot(
        ab.tp_count[star.ntpagb :],
        ab.nb.intershell[star.ntpagb :],
        c="C0",
        linewidth=3,
        alpha=0.5,
        label="MESA intershell",
    )
    plt.plot(
        ab.tp_count[star.ntpagb :],
        ab.nb.envelope[star.ntpagb :],
        c="C0",
        label="MESA envelope",
    )
    # plt.plot(ab.tp_count[star.ntpagb:], ab.nb.m_yield[star.ntpagb:])

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)

filtered = intershell.query("M1tp==2.5 and Z == 0.007 and last == 1")
plt.plot(
    filtered.ntp,
    93 * filtered.nb93 + 94 * filtered.nb94 + 95 * filtered.nb95,
    c="C1",
    linewidth=3,
    alpha=0.5,
    label="Monash intershell $z=0.007$",
)

with open("data/env_pd_df.pkl", "rb") as f:
    envelope = pickle.load(f)

filtered = envelope.query(
    "M_init == 2.5 and N_ov != 0 and element == 'nb' and Z == 0.007"
)
plt.plot(filtered["ntp"], filtered.massfrac, c="C1", label="Monash envelope $z=0.007$")

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)

filtered = intershell.query("M1tp==2.5 and Z == 0.0028 and last == 1")
plt.plot(
    filtered.ntp,
    93 * filtered.nb93 + 94 * filtered.nb94 + 95 * filtered.nb95,
    c="C2",
    linewidth=3,
    alpha=0.5,
    label="Monash intershell $z=0.0028$",
)

with open("data/env_pd_df.pkl", "rb") as f:
    envelope = pickle.load(f)

filtered = envelope.query(
    "M_init == 2.5 and N_ov != 0 and element == 'nb' and Z == 0.0028 and pmz==0.002"
)
plt.plot(filtered["ntp"], filtered.massfrac, c="C2", label="Monash envelope $z=0.0028$")

plt.yscale("log")

plt.text(
    0.05, 0.95, "$M = 2.5\\;M_\\odot$", transform=axs.transAxes, va="top", ha="left"
)

fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$X(\\textrm{Nb})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-check-nb.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in np.arange(2.5, 2.501, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    els = []
    plt.plot(
        ab.tp_count[star.ntpagb :],
        ab.nb.intershell[star.ntpagb :],
        c="C0",
        linewidth=3,
        alpha=0.5,
        label="MESA intershell",
    )
    plt.plot(
        ab.tp_count[star.ntpagb :],
        ab.nb.envelope[star.ntpagb :],
        c="C0",
        label="MESA envelope",
    )
    # plt.plot(ab.tp_count[star.ntpagb:], ab.nb.m_yield[star.ntpagb:])


with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)

filtered = intershell.query("M1tp==2.5 and Z == 0.0028 and last == 1")
for c, i in enumerate(range(90, 97)):
    plt.plot(
        filtered.ntp,
        i * filtered[f"zr{i}"],
        c=f"C{c}",
        linewidth=3,
        alpha=0.5,
        label=f"Zr{i}",
    )

plt.yscale("log")

plt.text(
    0.05, 0.95, "$M = 2.5\\;M_\\odot$", transform=axs.transAxes, va="top", ha="left"
)

fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$X(\\textrm{Nb})$")
# plt.savefig("/home/koen/LaTeX-setup/plots/w28-check-nb.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in np.arange(2.5, 2.501, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    els = []
    plt.plot(
        ab.tp_count[star.ntpagb :],
        ab.nb.intershell[star.ntpagb :],
        c="C0",
        linewidth=3,
        alpha=0.5,
        label="MESA intershell",
    )
    plt.plot(
        ab.tp_count[star.ntpagb :],
        ab.nb.envelope[star.ntpagb :],
        c="C0",
        label="MESA envelope",
    )
    # plt.plot(ab.tp_count[star.ntpagb:], ab.nb.m_yield[star.ntpagb:])

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)

filtered = intershell.query("M1tp==2.5 and Z == 0.007 and last == 1")
plt.plot(
    filtered.ntp,
    93 * filtered.nb93 + 94 * filtered.nb94 + 95 * filtered.nb95,
    c="C1",
    linewidth=3,
    alpha=0.5,
    label="Monash intershell $z=0.007$",
)

with open("data/env_pd_df.pkl", "rb") as f:
    envelope = pickle.load(f)

filtered = envelope.query(
    "M_init == 2.5 and N_ov != 0 and element == 'nb' and Z == 0.007"
)
plt.plot(filtered["ntp"], filtered.massfrac, c="C1", label="Monash envelope $z=0.007$")

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)

filtered = intershell.query("M1tp==2.5 and Z == 0.0028 and last == 1")
plt.plot(
    filtered.ntp,
    93 * filtered.nb93 + 94 * filtered.nb94 + 95 * filtered.nb95,
    c="C2",
    linewidth=3,
    alpha=0.5,
    label="Monash intershell $z=0.0028$",
)

with open("data/env_pd_df.pkl", "rb") as f:
    envelope = pickle.load(f)

filtered = envelope.query(
    "M_init == 2.5 and N_ov != 0 and element == 'nb' and Z == 0.0028 and pmz==0.002"
)
plt.plot(filtered["ntp"], filtered.massfrac, c="C2", label="Monash envelope $z=0.0028$")

plt.yscale("log")

plt.text(
    0.05, 0.95, "$M = 2.5\\;M_\\odot$", transform=axs.transAxes, va="top", ha="left"
)

fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$X(\\textrm{Nb})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-fix-nb.pgf", format="pgf")
plt.show()
plt.close()
# %%
import matplotlib.transforms as mtransforms

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
labels = []
for el in els:
    labels.append(el.capitalize())

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylim(0.1, 10)
plt.ylabel("$M_X / M_{X,M=2.5\\;M_\\odot,\\;Z=0.007}$")
plt.yscale("log")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w28-yield-all-diff-m2.5-fix-nb.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

i = 1e6
t_half = 1.61e6


def zr(i, t, t_half):
    return i * (1 / 2) ** (t / t_half)


t = np.linspace(0, 10e6, 300)
plt.plot(t, zr(i, t, t_half))
plt.show()
# %%


def dNdt(lam, N):
    return -lam * N


dt = 1000
N = 0
t_half = 1.61e6
lam = np.log(2) / t_half

ns = [N]
ts = [0]
for i in range(5000):
    N = N + dNdt(lam, N) * dt
    if i % 10 == 0:
        ts.append(ts[-1] + 10 * dt)
        ns.append(N)

    if i % 100 == 0 and i != 0:
        N += 0.01e6

plt.plot(ts[:-1], ns[1:])
i = 1e6
t_half = 1.61e6

dt = 1000
N = 0

ns = [N]
ts = [0]
for i in range(5000):
    if i % 10 == 0:
        ts.append(ts[-1] + 10 * dt)
        ns.append(N)

    if i % 100 == 0 and i != 0:
        N += 0.01e6

plt.plot(ts[:-1], ns[1:])
i = 1e6
t_half = 1.61e6


plt.show()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column, height=0.8), constrained_layout=True
)

ab = Abundances(None, df, method="tp offset", mass=m)
star = get_star(m=m)
final = []
z = []
els = []

star = get_star(m=m)

plt.plot(
    ab.time[star.ntpagb :] - ab.time[star.ntpagb],
    ab.zr93.intershell[star.ntpagb :],
    label="Intershell",
    linewidth=1,
)
plt.plot(
    ab.time[star.ntpagb :] - ab.time[star.ntpagb],
    ab.zr93.envelope[star.ntpagb :],
    label="Envelope without decay",
    linewidth=1,
)
plt.plot(
    ab.time[star.ntpagb :] - ab.time[star.ntpagb],
    ab.zr93.envelope_decay[star.ntpagb :],
    label="Envelope with decay",
    linewidth=1,
)

plt.plot(
    ab.time[star.ntpagb :] - ab.time[star.ntpagb],
    ab.zr93.envelope[star.ntpagb :] - ab.zr93.envelope_decay[star.ntpagb :],
    label="Decayed enevelope",
    linewidth=1,
)

plt.yscale("log")
plt.ylim(1e-10)
plt.xlabel("TPAGB age (yr)")
plt.ylabel("$X(^{93}\\textrm{Zr})$")

fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w28-envelope-decay.pgf", format="pgf")
plt.show()
plt.close()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in np.arange(2.5, 2.501, 0.1):
    ab = Abundances(None, df, method="tp offset", mass=m)
    star = get_star(m=m)
    final = []
    z = []
    els = []
    plt.plot(
        ab.tp_count[star.ntpagb :],
        ab.nb.intershell[star.ntpagb :],
        c="C0",
        linewidth=3,
        alpha=0.5,
        label="MESA intershell",
    )
    decay = ab.zr93.envelope[star.ntpagb :] - ab.zr93.envelope_decay[star.ntpagb :]

    plt.plot(
        ab.tp_count[star.ntpagb :],
        ab.nb.envelope[star.ntpagb :] + decay,
        c="C0",
        label="MESA envelope",
    )
    # plt.plot(ab.tp_count[star.ntpagb:], ab.nb.m_yield[star.ntpagb:])

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)

filtered = intershell.query("M1tp==2.5 and Z == 0.007 and last == 1")
plt.plot(
    filtered.ntp,
    93 * filtered.nb93 + 94 * filtered.nb94 + 95 * filtered.nb95,
    c="C1",
    linewidth=3,
    alpha=0.5,
    label="Monash intershell $z=0.007$",
)

with open("data/env_pd_df.pkl", "rb") as f:
    envelope = pickle.load(f)

filtered = envelope.query(
    "M_init == 2.5 and N_ov != 0 and element == 'nb' and Z == 0.007"
)
plt.plot(filtered["ntp"], filtered.massfrac, c="C1", label="Monash envelope $z=0.007$")

with open("data/intershell_pd_df.pkl", "rb") as f:
    intershell = pickle.load(f)

filtered = intershell.query("M1tp==2.5 and Z == 0.0028 and last == 1")
plt.plot(
    filtered.ntp,
    93 * filtered.nb93 + 94 * filtered.nb94 + 95 * filtered.nb95,
    c="C2",
    linewidth=3,
    alpha=0.5,
    label="Monash intershell $z=0.0028$",
)

with open("data/env_pd_df.pkl", "rb") as f:
    envelope = pickle.load(f)

filtered = envelope.query(
    "M_init == 2.5 and N_ov != 0 and element == 'nb' and Z == 0.0028 and pmz==0.002"
)
plt.plot(filtered["ntp"], filtered.massfrac, c="C2", label="Monash envelope $z=0.0028$")

plt.yscale("log")

plt.text(
    0.05, 0.95, "$M = 2.5\\;M_\\odot$", transform=axs.transAxes, va="top", ha="left"
)

fig.legend(loc="outside upper center", ncols=2)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Thermal pulse count")
plt.ylabel("$X(\\textrm{Nb})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-fix-nb-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

import matplotlib.transforms as mtransforms

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

        if el in ["nb"]:
            e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
            print(e)
            x = ab.zr93.envelope
            e += np.cumsum(x * ab.dm)[-1]
            print(e)

        elif el in ["zr"]:
            e = ab.__getattr__(ab.df.elements[el].key).m_yield[-1]
            x = ab.zr93.envelope
            e -= np.cumsum(x * ab.dm)[-1]
        else:
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
labels = []
for el in els:
    labels.append(el.capitalize())

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylim(0.1, 10)
plt.ylabel("$M_X / M_{X,M=2.5\\;M_\\odot,\\;Z=0.007}$")
plt.yscale("log")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w28-yield-all-diff-m2.5-fix-nb-3.pgf", format="pgf"
)
plt.show()
plt.close()
# %%

import matplotlib.transforms as mtransforms

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
        np.cumsum(np.array(final)),
        label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$",
        linewidth=1,
    )


m_yield_new = yields.query(
    "`Initial mass` == 2.5 and metallicity == 0.0028 and M_mix == 0.002"
)

# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.cumsum(np.array(m_yield_new["Mass(i)"])),
    label="$M=2.5\\;M_\\odot,\\;Z=0.0028,\\;M_\\textrm{mix} = 0.002$",
    linewidth=1,
)


m_yield_new = yields.query("`Initial mass` == 2.5 and metallicity == 0.007")
# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    np.cumsum(np.array(m_yield_new["Mass(i)"])),
    label="$M=2.5\\;M_\\odot,\\;Z=0.007$",
    linewidth=1,
)


fig.legend(loc="outside upper center", ncols=2)

labels = els
labels = []
for el in els:
    labels.append(el.capitalize())

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


# plt.ylim(0.1, 10)
plt.ylabel("$\sum M_X$ ($M_\\odot$)")
# plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-yield-cum.pgf", format="pgf")
plt.show()
plt.close()
# %%
import matplotlib.transforms as mtransforms

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
        (np.array(final) / np.sum(final))
        / (np.array(m_yield["Mass(i)"]) / np.sum(np.array(m_yield["Mass(i)"]))),
        label=f"MESA $M={m:.1f}\\;M_\\odot,\\;Z=0.00557$",
        linewidth=1,
    )


m_yield_new = yields.query(
    "`Initial mass` == 2.5 and metallicity == 0.0028 and M_mix == 0.002"
)

# plt.plot(m_yield["El"], m_yield["Mass(i)"])
plt.plot(
    m_yield["Z"],
    (np.array(m_yield_new["Mass(i)"]) / np.sum(np.array(m_yield_new["Mass(i)"])))
    / (np.array(m_yield["Mass(i)"]) / np.sum(np.array(m_yield["Mass(i)"]))),
    label="$M=2.5\\;M_\\odot,\\;Z=0.0028,\\;M_\\textrm{mix} = 0.002$",
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
labels = []
for el in els:
    labels.append(el.capitalize())

plt.xlabel("Element")
plt.xticks(z[::2], labels=labels[::2])

ax_t = axs.secondary_xaxis("top")
ax_t.set_xticks(z[1::2])
ax_t.set_xticklabels(labels[1::2])

for label in axs.get_xticklabels():
    label.set_verticalalignment("baseline")

for label in ax_t.get_xticklabels():
    label.set_verticalalignment("baseline")

for i, label in enumerate(axs.get_xticklabels()):
    label.set_y(-0.00 if i % 2 == 0 else -0.03)


for i, tick in enumerate(ax_t.xaxis.get_major_ticks()):
    tick.tick2line.set_markersize(7 if i % 2 else 0)

    label = tick.label2

    offset = 10 if i % 2 else 3

    label.set_transform(
        ax_t.get_xaxis_transform()
        + mtransforms.ScaledTranslation(
            0,
            offset / 72,
            fig.dpi_scale_trans,
        )
    )


ticks = axs.xaxis.get_major_ticks()

for i, tick in enumerate(ticks):
    length = 7 if i % 2 else 0

    tick.tick1line.set_markersize(length)
    tick.tick2line.set_markersize(length)

for i in z[::2]:
    axs.axvline(i, color="C9", linewidth=0.75 / 2, zorder=-1)


plt.ylim(0.1, 10)
plt.ylabel("$M_X / M_{X,M=2.5\\;M_\\odot,\\;Z=0.007}$")
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w28-yield-mass-correction.pgf", format="pgf")
plt.show()
plt.close()
# %%
