import time
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
from scripts.general_utils.asplund import Element, Asplund

plt.cplot = cplot

# %%

df = AbundanceTables()
ab = Abundances(None, df, mass=2.8, sampling=100)


intershell = ab.compute_all_intershell()
envelope = ab.compute_all_envelope_abundances(intershell)
yields = np.cumsum(envelope * ab.dm[:, None], axis=0)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

final_env = []
final_inter = []
final_yield = []

counter = list(range(0, 81))

for _, i in enumerate(counter):
    final_env.append(envelope[:, i][-1])
    final_inter.append(intershell[:, i][-1])
    final_yield.append(yields[-1, i])

plt.plot(ab.elements_mass, np.array(final_yield) / np.array(final_yield))


df = AbundanceTables()
ab = Abundances(None, df, mass=2.8, sampling=10)

intershell = ab.compute_all_intershell()
envelope = ab.compute_all_envelope_abundances(intershell)
yields = np.cumsum(envelope * ab.dm[:, None], axis=0)


final_env = []
final_inter = []
final_yield_2 = []

counter = list(range(0, 81))

for _, i in enumerate(counter):
    final_env.append(envelope[:, i][-1])
    final_inter.append(intershell[:, i][-1])
    final_yield_2.append(yields[-1, i])

plt.plot(ab.elements_mass, np.array(final_yield_2) / np.array(final_yield))

df = AbundanceTables()


ab = Abundances(None, df, method="tp offset", mass=2.8)
start = time.time()
star = get_star(m=2.8)
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
    e = ab.__getattr__(ab.df.elements[el].key).m_yield
    e = e[-1]
    final.append(e)
    data = ab.df.envelope[ab.df.envelope["element"] == el]
    z.append(int(np.array(data["elemental_mass"])[0]))
    # z.append(ab.df.envelope.query(f"element == {el}")["elemental_mass"])
    els.append(el)

end = time.time()
print("naive = ", end - start)

plt.plot(
    z,
    np.array(final) / np.array(final_yield),
    label=f"MESA $M={2.5:.1f}\\;M_\\odot,\\;Z=0.00557$",
    linewidth=1,
)


plt.xlabel("Element")
plt.ylabel("$X_\\textrm{f}$")
# plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
plt.yscale("log")
# plt.savefig("/home/koen/LaTeX-setup/plots/w28-envelope-diff.pgf", format="pgf")
plt.show()
plt.close()
# %%

df = AbundanceTables()
ab = Abundances(None, df, mass=2.8, sampling=100)
intershell = ab.compute_all_intershell()
envelope = ab.compute_all_envelope_abundances(intershell)
yields = np.cumsum(envelope * ab.dm[:, None], axis=0)

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)
df = AbundanceTables()
ab = Abundances(None, df, mass=2.8, sampling=1)

intershell = ab.compute_all_intershell()
envelope = ab.compute_all_envelope_abundances(intershell)
yields = np.cumsum(envelope * ab.dm[:, None], axis=0)

final_env = []
final_inter = []
final = []

counter = list(range(0, 81))

for _, i in enumerate(counter):
    final_env.append(envelope[:, i][-1])
    final_inter.append(intershell[:, i][-1])
    final.append(yields[-1, i])


for sampling in np.logspace(np.log10(1.95), 3, 10):
    sampling = round(sampling)
    df = AbundanceTables()
    ab = Abundances(None, df, mass=2.8, sampling=sampling)

    intershell = ab.compute_all_intershell()
    envelope = ab.compute_all_envelope_abundances(intershell)
    yields = np.cumsum(envelope * ab.dm[:, None], axis=0)

    final_env = []
    final_inter = []
    final_yield = []

    counter = list(range(0, 81))

    for _, i in enumerate(counter):
        final_env.append(envelope[:, i][-1])
        final_inter.append(intershell[:, i][-1])
        final_yield.append(yields[-1, i])

    plt.plot(
        ab.elements_mass,
        np.abs(np.array(final_yield) - np.array(final)) / np.array(final),
        label=f"ss=1/{int(sampling)}",
    )

fig.legend(loc="outside upper center", ncols=5)


labels = ab.elements_name
for i, _ in enumerate(labels):
    labels[i] = labels[i].capitalize()

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


plt.ylabel("$|M(X)_\\textrm{f,ss} - M(X)_\\textrm{f,no ss}| / M(X)_\\textrm{f,no ss}$")
# plt.axhline(1, c="C9", linewidth=0.75, zorder=-10)
plt.yscale("log")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-effect-of-subsampling.pgf", format="pgf")
plt.show()
plt.close()
# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

times = []
sss = np.logspace(np.log10(1.95), 3, 100)

for i, sampling in enumerate(sss):
    print(i)
    start = time.time()
    sampling = round(sampling)
    df = AbundanceTables()
    ab = Abundances(None, df, mass=2.8, sampling=sampling)

    intershell = ab.compute_all_intershell()
    envelope = ab.compute_all_envelope_abundances(intershell)
    yields = np.cumsum(envelope * ab.dm[:, None], axis=0)

    final_env = []
    final_inter = []
    final_yield = []

    counter = list(range(0, 81))

    for _, i in enumerate(counter):
        final_env.append(envelope[:, i][-1])
        final_inter.append(intershell[:, i][-1])
        final_yield.append(yields[-1, i])

    end = time.time()
    times.append(end - start)


axs.spines[["right", "top"]].set_visible(False)
plt.xlabel("subsampling factor (1/ss)")
plt.ylabel("Time in seconds for 100 models")
plt.savefig("/home/koen/LaTeX-setup/plots/w29-subsampling-time.pgf", format="pgf")
plt.show()
plt.close()
