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

sys.path.insert(1, "/home/koen/master-internship/")
MASTER = "/home/koen/master-internship/mesa-models/"

import mesa_reader as mr
from scripts.general_utils.mesa_grid_2 import MesaGrid, SimpleBinary

# %%

grid = MesaGrid(f"{MASTER}/grid-broad-R-2026-09-27")
df = AbundanceTables()

# %%

for model in grid.models:
    plt.plot(model.age, model.rl_1)
    # plt.plot(model.age, model.R)

star = get_star(m=2.2)
plt.plot(star.age, 10**star.log_R, c="C9", zorder=-1)
plt.show()
# %%

df = AbundanceTables()

for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue
    print(m.params["m"])
    mass = m.params["m"]
    ab = Abundances(model=m, df=df, sampling=1)
    m.ab = ab.iron_abundance
    m.elements_mass = ab.elements_mass
    m.elements_name = ab.elements_name
    m.t_av = ab.t_av
    m.m_dup_av = ab.m_dup_av


def get_ls(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in ["sr", "y", "zr"]:
            fe += r.ab[i]

    return fe / 3


def get_hs(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in [
            "ba",
            "la",
            "ce",
            "nd",
        ]:
            fe += r.ab[i]

    return fe / 4


def get_s(r):
    if r.envelope_mass[-1] > 0.02:
        return np.nan
    fe = 0
    for i, name in enumerate(r.elements_name):

        if name in ["ba", "la", "ce", "nd", "sr", "y", "zr"]:
            fe += r.ab[i]

    return fe / 7


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

per = []
s = []
for m in grid.models:

    per.append(m.params["R"])
    s.append(get_s(m))

per = np.array(per)
s = np.array(s)
idx = np.argsort(per)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)

plt.axhline(0.25, c="C9", linewidth=0.75, zorder=-10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-s-fe-RL.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

per = []
s = []
for m in grid.models:

    per.append(m.params["R"])
    s.append(get_hs(m) - get_ls(m))

per = np.array(per)
s = np.array(s)
idx = np.argsort(per)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[hs/ls]")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-hs-ls-RL.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

per = []
s = []
for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue

    per.append(m.m_dup_av)
    s.append(get_s(m))
plt.scatter(per, s, s=10, label=f"$M_\\textrm{{TPAGB}} = {mass:.1f}\\;M_\\odot$")

fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\langle M_\\textrm{DUP} \\rangle$ ($M_\\odot$)")
plt.ylabel("[s/Fe]")
# plt.xscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w30-s-fe-m_dup.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


i = 0
ls = []
for m in grid.models[::-1]:
    if m.params["R"] > 310 or m.params["R"] < 290:
        continue
    if m.envelope_mass[-1] > 0.01:
        continue

    ab = Abundances(model=m, df=df, save_full=True, sampling=1)
    (l1,) = plt.plot(
        ab.time,
        ab.envelope[:, 52],
        label=f"$R_\\textrm{{RL}}={m.params["R"]:.0f}\\;R_\\odot$",
        c=f"C{i}",
        zorder=-i,
    )
    (l2,) = plt.plot(ab.time, ab.intershell[:, 52], c=f"C{i}", linestyle="--")
    plt.scatter(
        ab.time[ab.simple_end_idx], ab.envelope[ab.simple_end_idx, 52], zorder=1000
    )
    i += 1
    ls.append(l1)
    ls.append(l2)

ab = Abundances(model=None, df=df, mass=2.2, m_acc=1, save_full=True, sampling=1)
(l3,) = plt.plot(
    ab.time,
    ab.envelope[:, 52],
    label=f"single",
    zorder=-1,
    c="C9",
    linewidth=3,
    alpha=0.5,
)
plt.plot(
    ab.time,
    ab.intershell[:, 52],
    zorder=-1,
    c="C9",
    linewidth=3,
    alpha=0.5,
    linestyle="--",
)
fig.legend(
    loc="outside upper center",
    ncols=3,
    handles=[ls[0], ls[2], ls[0], ls[1], l3],
    labels=[
        "$R_\\textrm{{RL}}=291\\;R_\\odot$",
        "$R_\\textrm{{RL}}=305\\;R_\\odot$",
        "Envelope",
        "Intershell",
        "Single star",
    ],
)
plt.xlim(920224429.7471193, 922619125.6257955)
plt.ylim(2.3597730260456065e-09, 6.826683073898937e-05)
plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel("$X(\\textrm{Ba})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-show-wrong-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

for m in grid.models:
    if m.params["R"] > 400 or m.params["R"] < 184:
        continue
    if m.envelope_mass[-1] > 0.01:
        continue

    ab = Abundances(model=m, df=df, save_full=True, sampling=1)
    print(ab.elements_name[52])
    plt.plot(ab.time, np.cumsum(ab.m_dup), label=f"{m.params["R"]:.0f}")
    plt.scatter(
        ab.time[ab.simple_end_idx],
        np.cumsum(ab.m_dup)[ab.simple_end_idx],
        label=f"{m.params["R"]:.0f}",
    )

ab = Abundances(model=None, df=df, mass=2.2, m_acc=1, save_full=True, sampling=1)
plt.plot(
    ab.time,
    np.cumsum(ab.m_dup),
    label=f"single",
    zorder=-1,
    c="C9",
    linewidth=2,
    alpha=0.5,
)
plt.legend()
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

rs = []
per = []
s = []
for m in grid.models:

    rs.append(m.params["R"])
    per.append(get_s(m))
    s.append(get_hs(m) - get_ls(m))

rs = np.array(rs)
per = np.array(per)
s = np.array(s)
idx = np.argsort(rs)

rs = rs[idx]
per = per[idx]
s = s[idx]
c = plt.cplot(per, s, rs, linewidth=1, vmin=200, vmax=1200)

plt.colorbar(c, extend="both", label="$R_\\textrm{RL}$ ($R_\\odot$)")

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("[s/Fe]")
plt.ylabel("[hs/ls]")
# plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-hs-ls-s-RL.pgf", format="pgf", dpi=600)
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

rs = []
per = []
s = []
for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue

    rs.append(m.params["R"])
    per.append(m.m_dup_av)
    s.append(get_s(m))
plt.scatter(
    per,
    s,
    c=rs,
    s=10,
    label=f"$M_\\textrm{{TPAGB}} = {mass:.1f}\\;M_\\odot$",
    cmap="viridis",
    vmin=280,
    vmax=300,
)

fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\langle M_\\textrm{DUP} \\rangle$ ($M_\\odot$)")
plt.ylabel("[s/Fe]")
# plt.xscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w30-s-fe-m_dup.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue

    ab = Abundances(model=m, df=df, save_full=True, sampling=1)
    print(ab.elements_name[52])
    plt.plot(ab.time[ab.simple_end_idx :], ab.tp_count[ab.simple_end_idx :])
    plt.scatter(
        ab.time[ab.simple_end_idx],
        ab.tp_count[ab.simple_end_idx],
        zorder=11,
        marker=".",
        s=75,
    )
    plt.scatter(
        ab.time[ab.simple_end_idx],
        ab.tp_count[ab.simple_end_idx],
        c="w",
        zorder=10,
        marker=".",
        s=150,
    )

plt.xlim(918873256.7322208, 922653520.8768578)
star = get_star(m=2.2)
plt.plot(star.age, star.TP_count, c="C9", linewidth=3, alpha=0.5)

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel("TP count")
plt.savefig("/home/koen/LaTeX-setup/plots/w-30-show-wrong-2.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

rs = []
per = []
s = []

for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue

    rs.append(m.params["R"])
    per.append(m.m_dup_av)
    s.append(get_s(m))

rs = np.array(rs)
per = np.array(per)
s = np.array(s)
idx = np.argsort(rs)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)
plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
axs.spines[["right", "top"]].set_visible(False)
plt.axhline(0.25, color="C9", linewidth=0.75, zorder=-1)
plt.xlabel("$\\langle M_\\textrm{DUP}\\rangle$ ($M_\\odot$)")
plt.ylabel("[s/Fe]")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-s-dup-2.pgf", format="pgf")
plt.show()
plt.close()


# %%
def roche_lobe(q):
    """
    Eggleton's (1983) formula for the relative roche-lobe radius, R_L/a.

        Args:
        q = M_star/M_companion
    """
    q13 = q ** (1.0 / 3.0)
    q23 = q13**2
    rl = 0.49 * q23 / (0.6 * q23 + np.log(1 + q13))
    return rl


for model in grid.models:
    pass

q = star.mass / model.sb.m2
RL = model.sb.a * roche_lobe(q)

plt.plot(star.age, RL)
plt.plot(model.age, model.rl_1)
plt.show()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


i = 0
ls = []
for m in grid.models[::-1]:
    if m.params["R"] > 310 or m.params["R"] < 290:
        continue
    if m.envelope_mass[-1] > 0.01:
        continue

    ab = Abundances(model=m, df=df, save_full=True, sampling=1)
    (l1,) = plt.plot(
        ab.time,
        ab.envelope[:, 52],
        label=f"$R_\\textrm{{RL}}={m.params["R"]:.0f}\\;R_\\odot$",
        c=f"C{i}",
        zorder=-i,
    )
    (l2,) = plt.plot(ab.time, ab.intershell[:, 52], c=f"C{i}", linestyle="--")
    plt.scatter(
        ab.time[ab.simple_end_idx], ab.envelope[ab.simple_end_idx, 52], zorder=1000
    )
    i += 1
    ls.append(l1)
    ls.append(l2)

ab = Abundances(model=None, df=df, mass=2.2, m_acc=1, save_full=True, sampling=1)
(l3,) = plt.plot(
    ab.time,
    ab.envelope[:, 52],
    label=f"single",
    zorder=-1,
    c="C9",
    linewidth=3,
    alpha=0.5,
)
plt.plot(
    ab.time,
    ab.intershell[:, 52],
    zorder=-1,
    c="C9",
    linewidth=3,
    alpha=0.5,
    linestyle="--",
)
fig.legend(
    loc="outside upper center",
    ncols=3,
    handles=[ls[0], ls[2], ls[0], ls[1], l3],
    labels=[
        "$R_\\textrm{{RL}}=291\\;R_\\odot$",
        "$R_\\textrm{{RL}}=305\\;R_\\odot$",
        "Envelope",
        "Intershell",
        "Single star",
    ],
)
plt.xlim(920224429.7471193, 922619125.6257955)
plt.ylim(2.3597730260456065e-09, 6.826683073898937e-05)
plt.yscale("log")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("Star age (yr)")
plt.ylabel("$X(\\textrm{Ba})$")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-show-right-1.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

rs = []
per = []
s = []

for m in grid.models:
    if m.envelope_mass[-1] > 0.01:
        continue

    rs.append(m.params["R"])
    per.append(m.m_dup_av)
    s.append(get_s(m))

rs = np.array(rs)
per = np.array(per)
s = np.array(s)
idx = np.argsort(rs)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)
plt.axvspan(0.0035, 0.01, alpha=0.2, color="C9")
plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$\\langle M_\\textrm{DUP}\\rangle$ ($M_\\odot$)")
plt.ylabel("[s/Fe]")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-s-dup.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

per = []
s = []
for m in grid.models:

    per.append(m.params["R"])
    s.append(get_s(m))

per = np.array(per)
s = np.array(s)
idx = np.argsort(per)

per = per[idx]
s = s[idx]


plt.plot(per, s, c="k", linewidth=1)
plt.scatter(per, s, s=75, c="k", marker=".", zorder=20)
plt.scatter(per, s, s=150, c="w", marker=".", zorder=10)

plt.axhline(0.25, c="C9", linewidth=0.75, zorder=-10)

plt.title("$M_\\textrm{TPAGB,i} = 2.2\\;M_\\odot$, $q=0.6$, $\\epsilon=0.25$")
fig.legend(loc="outside upper center", ncols=2)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("[s/Fe]")
plt.xscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-s-fe-RL-fix.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

suc_rs = []
fail_rs = []
suc_ys = []
fail_ys = []

for model in grid.models:
    if model.envelope_mass[-1] > 0.01:
        fail_rs.append(model.params["R"])
        arg = np.argmax(model.lg_mstar_dot_1)
        fail_ys.append(1 / (10 ** model.lg_mstar_dot_1[arg] / model.envelope_mass[arg]))

    else:
        suc_rs.append(model.params["R"])
        arg = np.argmax(model.lg_mstar_dot_1)
        suc_ys.append(1 / (10 ** model.lg_mstar_dot_1[arg] / model.envelope_mass[arg]))

suc_rs = np.array(suc_rs)
fail_rs = np.array(fail_rs)
suc_ys = np.array(suc_ys)
fail_ys = np.array(fail_ys)


indx_fail = np.argsort(fail_rs)
fail_rs = fail_rs[indx_fail]
fail_ys = fail_ys[indx_fail]

indx_suc = np.argsort(suc_rs)
suc_rs = suc_rs[indx_suc]
suc_ys = suc_ys[indx_suc]

plt.scatter(suc_rs, suc_ys, zorder=10, c="w", marker=".", s=150)
plt.scatter(suc_rs, suc_ys, zorder=11, c="C2", marker=".", s=75)
plt.plot(suc_rs, suc_ys, c="C2")
plt.scatter(fail_rs, fail_ys, zorder=10, c="w", marker=".", s=150)
plt.scatter(fail_rs, fail_ys, zorder=11, c="C3", marker=".", s=75)
plt.plot(fail_rs, fail_ys, c="C3")


axs.annotate(
    "Failing models",
    xy=(170, 20),
    xycoords="data",
    xytext=(0, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)

axs.annotate(
    "Succesful models",
    xy=(600, 20),
    xycoords="data",
    xytext=(-50, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)


plt.yscale("log")
plt.xscale("log")

plt.xlim(100)
axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{RL,i}$ ($R_\\odot$)")
plt.ylabel("$M_\\textrm{env} / \\textrm{max}(\\dot{M})$ (yr$^{-1}$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-failing-models-1.pgf", format="pgf")
plt.show()
plt.close()

# %%
model.bulk_names

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

suc_rs = []
fail_rs = []
suc_ys = []
fail_ys = []

for model in grid.models:
    if model.envelope_mass[-1] > 0.01:

        fail_rs.append(model.sb.a[np.isfinite(model.sb.a)][0])
        arg = np.argmax(model.lg_mstar_dot_1)
        fail_ys.append(1 / (10 ** model.lg_mstar_dot_1[arg] / model.envelope_mass[arg]))

    else:
        suc_rs.append(model.sb.a[np.isfinite(model.sb.a)][0])
        arg = np.argmax(model.lg_mstar_dot_1)
        suc_ys.append(1 / (10 ** model.lg_mstar_dot_1[arg] / model.envelope_mass[arg]))

suc_rs = np.array(suc_rs)
fail_rs = np.array(fail_rs)
suc_ys = np.array(suc_ys)
fail_ys = np.array(fail_ys)


indx_fail = np.argsort(fail_rs)
fail_rs = fail_rs[indx_fail]
fail_ys = fail_ys[indx_fail]

indx_suc = np.argsort(suc_rs)
suc_rs = suc_rs[indx_suc]
suc_ys = suc_ys[indx_suc]

plt.scatter(suc_rs, suc_ys, zorder=10, c="w", marker=".", s=150)
plt.scatter(suc_rs, suc_ys, zorder=11, c="C2", marker=".", s=75)
plt.plot(suc_rs, suc_ys, c="C2")
plt.scatter(fail_rs, fail_ys, zorder=10, c="w", marker=".", s=150)
plt.scatter(fail_rs, fail_ys, zorder=11, c="C3", marker=".", s=75)
plt.plot(fail_rs, fail_ys, c="C3")


axs.annotate(
    "Failing models",
    xy=(450, 20),
    xycoords="data",
    xytext=(0, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)

axs.annotate(
    "Succesful models",
    xy=(1700, 20),
    xycoords="data",
    xytext=(-50, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)


plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$a_\\textrm{i}$ ($R_\\odot$)")
plt.ylabel("$M_\\textrm{env} / \\textrm{max}(\\dot{M})$ (yr$^{-1}$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w30-failing-models-2.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

suc_rs = []
fail_rs = []
suc_ys = []
fail_ys = []

for model in grid.models:
    arg_M = np.argmax(10**model.lg_mstar_dot_1 / model.star_mass)
    M = np.max(10**model.lg_mstar_dot_1 / model.star_mass)
    arg_a = np.argmax(
        np.abs(np.diff(model.binary_separation) / np.diff(model.age))
        / model.binary_separation[1:]
    )
    a = np.max(
        np.abs(np.diff(model.binary_separation) / np.diff(model.age))
        / model.binary_separation[1:]
    )

    res = [M, a]
    arg_res = [arg_M, arg_a]

    arg_max = np.argmax(res)
    print(res[arg_max])
    y = res[arg_max] * model.period_days[arg_res[arg_max]] / 365

    if model.envelope_mass[-1] > 0.01:

        fail_rs.append(model.sb.a[np.isfinite(model.sb.a)][0])
        fail_ys.append(y)

    else:
        suc_rs.append(model.sb.a[np.isfinite(model.sb.a)][0])
        suc_ys.append(y)

suc_rs = np.array(suc_rs)
fail_rs = np.array(fail_rs)
suc_ys = np.array(suc_ys)
fail_ys = np.array(fail_ys)


indx_fail = np.argsort(fail_rs)
fail_rs = fail_rs[indx_fail]
fail_ys = fail_ys[indx_fail]

indx_suc = np.argsort(suc_rs)
suc_rs = suc_rs[indx_suc]
suc_ys = suc_ys[indx_suc]

plt.scatter(suc_rs, suc_ys, zorder=10, c="w", marker=".", s=150)
plt.scatter(suc_rs, suc_ys, zorder=11, c="C2", marker=".", s=75)
plt.plot(suc_rs, suc_ys, c="C2")
plt.scatter(fail_rs, fail_ys, zorder=10, c="w", marker=".", s=150)
plt.scatter(fail_rs, fail_ys, zorder=11, c="C3", marker=".", s=75)
plt.plot(fail_rs, fail_ys, c="C3")


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$a_\\textrm{i}$ ($R_\\odot$)")
plt.ylabel(
    "$\\textrm{max}(|\\dot{M}_\\textrm{d} / M_\\textrm{d}|,\\; |\\dot{a}/ a|) \\cdot P$"
)
plt.savefig("/home/koen/LaTeX-setup/plots/w30-failing-models-3.pgf", format="pgf")
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

suc_rs = []
fail_rs = []
suc_ys = []
fail_ys = []

grid_sorted = []
rs = []
for model in grid.models:
    rs.append(model.params["R"])
    grid_sorted.append(model)

rs = np.array(rs)
grid_sorted = np.array(grid_sorted)

sort_idx = np.argsort(rs)
grid_sorted = grid_sorted[sort_idx]

for model in grid_sorted[::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.R,
            model.min_T,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c2 = plt.cplot(
            model.R[arg_start:arg_end],
            model.min_T[arg_start:arg_end],
            np.log10(model.envelope_mass[arg_start:arg_end]),
            cmap="Greens",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )

plt.colorbar(c1, label="log$(M_\\textrm{env} / M_\\odot)$", pad=0.01, aspect=50)
cbar = plt.colorbar(c2, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{star}$ ($R_\\odot$)")
plt.ylabel("min$(T)$ (K)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w30-failing-models-4.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()


# %%

model.bulk_names
# %%

for model in grid_sorted:
    if model.envelope_mass[-1] > 0.01:
        plt.plot(model.age, model.rl_1, c="C3")
    else:
        plt.plot(model.age, model.rl_1, c="C2")

star = get_star(m=2.2)
plt.plot(star.age, 10**star.log_R, c="C9", zorder=-1)
plt.show()
# %%

for i, model in enumerate(grid_sorted[:9]):
    plt.plot(model.age - 0.146e8, model.binary_separation, c=f"C{i}")
    plt.plot(star.age, model.sb.a, c=f"C{i}")

star = get_star(m=2.2)
plt.plot(star.age, 10**star.log_R, c="C9", zorder=-1)
plt.show()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

suc_rs = []
fail_rs = []
suc_ys = []
fail_ys = []

grid_sorted = []
rs = []
for model in grid.models:
    rs.append(model.params["R"])
    grid_sorted.append(model)

rs = np.array(rs)
grid_sorted = np.array(grid_sorted)

sort_idx = np.argsort(rs)
grid_sorted = grid_sorted[sort_idx]

for model in grid_sorted[36:][::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.R,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c3 = plt.cplot(
            model.R,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Blues",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
for model in grid_sorted[:36][::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.R,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c2 = plt.cplot(
            model.R,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Greens",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )

plt.colorbar(c1, label="log$(M_\\textrm{env} / M_\\odot)$", pad=0.01, aspect=50)
cbar = plt.colorbar(c3, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])
cbar = plt.colorbar(c2, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{star}$ ($R_\\odot$)")
plt.ylabel("$\\dot{M} / \\dot{M}_\\textrm{crit}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w30-failing-models-5.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

suc_rs = []
fail_rs = []
suc_ys = []
fail_ys = []

grid_sorted = []
rs = []
for model in grid.models:
    rs.append(model.params["R"])
    grid_sorted.append(model)

rs = np.array(rs)
grid_sorted = np.array(grid_sorted)

sort_idx = np.argsort(rs)
grid_sorted = grid_sorted[sort_idx]

for model in grid_sorted[36:][::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c3 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Blues",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
for model in grid_sorted[:36][::-1]:
    if model.envelope_mass[-1] > 0.01:
        c1 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Reds",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )
    else:
        try:
            arg_end = np.where(model.R > model.rl_1)[0][-1]
            arg_start = np.where(model.R > model.rl_1)[0][0]
        except IndexError:
            continue
        if len(model.envelope_mass[arg_start:arg_end]) == 0:
            continue
        c2 = plt.cplot(
            model.envelope_mass,
            10**model.lg_mstar_dot_1 / model.quasi_adiabatic_Mdot,
            np.log10(model.envelope_mass),
            cmap="Greens",
            vmin=-3,
            vmax=np.log10(1.7),
            linewidth=1,
        )

plt.colorbar(c1, label="log$(M_\\textrm{env} / M_\\odot)$", pad=0.01, aspect=50)
cbar = plt.colorbar(c3, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])
cbar = plt.colorbar(c2, pad=0.01, aspect=50)
cbar.set_ticks(ticks=[], labels=[])

plt.xscale("log")
plt.yscale("log")

axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("$R_\\textrm{star}$ ($R_\\odot$)")
plt.ylabel("$\\dot{M} / \\dot{M}_\\textrm{crit}$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w30-failing-models-6.pgf", format="pgf", dpi=600
)
plt.show()
plt.close()
