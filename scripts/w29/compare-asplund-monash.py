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
from scripts.general_utils.m_dup import (
    compute_m_DUP,
    AbundanceTables,
    Abundances,
    MonashModel,
)
from scripts.general_utils.asplund import Element, Asplund

plt.cplot = cplot
# %%

with open(f"data/env_pd_df.pkl", "rb") as f:
    env = pickle.load(f)

asplund = Asplund(z=0.014)

zs = []
massfracs = []
els = []
for Z, element in asplund.elements.items():
    zs.append(Z)
    els.append(element.name)
    massfracs.append(element.massfrac)

plt.plot(zs, massfracs)


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


interesting = env[env["pmz"] == 2e-3]
interesting = interesting[interesting["N_ov"] != 0]

zs = np.unique(interesting["Z"])

table = []
elements = None

dat = interesting[interesting["Z"] == 0.014]

mass = np.unique(dat["M_init"].astype(np.float64))
ind = np.argmin(np.abs(mass - 2.5))
data = dat[dat["M_init"].astype(np.float64) == mass[ind]]

# preserve the order from the original file
current_elements = [
    el for el in data["element"].drop_duplicates() if el not in ["p", "pm", "tc", "po"]
]

# use the ordering from the first metallicity
if elements is None:
    elements = current_elements

el_init = []
z = []
for element in elements:
    if element in ["tc", "po"]:
        continue
    el = data[data["element"] == element]
    el_init.append(float(el["massfrac"].iloc[0]))
    z.append(el["elemental_mass"].iloc[0])


print(z)
table = np.array(table)

# plot each metallicity
plt.plot(z, el_init)

plt.show()
plt.close()
# %%
