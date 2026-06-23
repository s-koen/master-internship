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
mesa = mr.MesaData(f"{MASTER}/standard-2msun-v3/LOGS/TPAGB/history.data")
# %%
mesa_eagb = mr.MesaData(f"{MASTER}/standard-2msun-v3/LOGS/EAGB/history.data")
# %%


import re
import pandas as pd


def parse_surf_file(path):

    header_regex = re.compile(r"Initial mass =\s*([\d.]+).*M_mix =\s*([\d.E+-]+)")

    tp_regex = re.compile(
        r"#\s+(\d+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)"
    )

    abundance_regex = re.compile(
        r"^\s*([a-z]{1,2})\s+\d+\s+[\d.E+-]+\s+[\d.E+-]+\s+([\d.E+-]+)\s+[\d.E+-]+\s+([\d.E+-]+)",
        re.IGNORECASE,
    )

    rows = []

    # state
    current_mass = None
    current_mmix = None
    current_tp_meta = None
    in_abundance_block = False
    last_tp = None
    final = False

    with open(path) as f:
        for line in f:

            # -------------------------
            # NEW MODEL (hard reset)
            # -------------------------
            h = header_regex.search(line)
            if h:
                current_mass = float(h.group(1))
                current_mmix = float(h.group(2))

                current_tp_meta = None
                in_abundance_block = False

                last_tp = None
                continue

            # -------------------------
            # TP HEADER
            # -------------------------
            t = tp_regex.match(line)
            if t:
                tp = int(t.group(1))

                if tp == last_tp:
                    continue

                last_tp = tp

                current_tp_meta = {
                    "M_init": current_mass,
                    "M_mix": current_mmix,
                    "TP": tp,
                    "Mass": float(t.group(2)),
                    "Mcore": float(t.group(3)),
                    "Menv": float(t.group(4)),
                    "logL": float(t.group(5)),
                }

                in_abundance_block = False
                continue

            # -------------------------
            # ABUNDANCE BLOCK START
            # -------------------------
            if line.strip().startswith("# El") and "X(i)" in line:
                in_abundance_block = True
                continue

            # -------------------------
            # ABUNDANCE BLOCK END MARKERS
            # -------------------------
            if line.startswith("# Elemental abundance ratios"):
                in_abundance_block = False
                continue

            if line.startswith("# Initial abundances"):
                in_abundance_block = False
                final = False
                continue

            if line.startswith("# Final abundances"):
                in_abundance_block = False
                final = True
                continue

            # -------------------------
            # ABUNDANCES
            # -------------------------

            if final and current_tp_meta:
                current_tp_meta["TP"] += 1
                final = False

            if in_abundance_block and current_tp_meta is not None:
                a = abundance_regex.match(line)
                if a:
                    rows.append(
                        {
                            **current_tp_meta,
                            "element": a.group(1),
                            "XFe": float(a.group(2)),
                            "massfrac": float(a.group(3)),
                        }
                    )

    return pd.DataFrame(rows)


# %%
df = parse_surf_file("scripts/w21/surf_z014.dat")

# %%
df2 = df[(df.M_init == 2.0) & (df.M_mix == 2e-3)]  # adjust if needed
# %%
df2
# %%
c = df2[df2.element == "c"][["TP", "massfrac"]].rename(columns={"massfrac": "C"})
o = df2[df2.element == "o"][["TP", "massfrac"]].rename(columns={"massfrac": "O"})
# %%
plt.plot(c.TP, np.array(c.C) / np.array(o.O) * 16 / 12)
plt.show()

# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.xlabel("Thermal Pulse count")
plt.ylabel("C/O-number ratio")
pulse = []
co = []
for i in range(len(mesa.star_age)):
    if mesa.TP_count[i] not in pulse:
        pulse.append(mesa.TP_count[i])
        co.append(mesa.surface_c12[i] / mesa.surface_o16[i] * 16 / 12)

plt.plot(pulse, co, ".-", label="MESA")
plt.plot(c.TP, np.array(c.C) / np.array(o.O) * 16 / 12, ".-", label="Karakas (2016)")

fig.legend(loc="outside upper center", ncols=2)
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-agb-nucleosynthis-compare-mesa.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
print(mesa.bulk_names)
# %%
df_core = df.drop_duplicates(["M_init", "TP"])
# %%
df_core = df_core[(df_core.M_init == 2.0) & (df_core.M_mix == 1e-3)]

# %%
df_core
# %%
c = df2[df2.element == "c"]
o = df2[df2.element == "o"]

# %%

pulse = []
core_mass = []
for i in range(len(mesa.star_age)):
    if mesa.TP_count[i] not in pulse:
        pulse.append(mesa.TP_count[i])
        core_mass.append(mesa.he_core_mass[i])

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(c.TP[:-1], c.Mcore[:-1], ".-", label="Karakas (2016)")
plt.plot(pulse, core_mass, ".-", label="MESA")


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("Thermal Pulse count")
plt.ylabel("He-core mass ($M_\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-agb-nucleosynthis-compare-core-mass.pgf",
    format="pgf",
)
plt.show()
plt.close()

# %%

pulse = []
star_mass = []
for i in range(len(mesa.star_age)):
    if mesa.TP_count[i] not in pulse:
        pulse.append(mesa.TP_count[i])
        star_mass.append(mesa.star_mass[i])

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(pulse, star_mass, ".-", label="MESA")
plt.plot(c.TP[:-1], c.Mass[:-1], ".-", label="Karakas (2016)")


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("Thermal Pulse count")
plt.ylabel("He-core mass ($M_\odot$)")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-agb-nucleosynthis-compare-star-mass.pgf",
    format="pgf",
)
plt.show()
plt.close()


# %%

nugrid = mr.MesaData(f"{MASTER}nugrid/M2z0.01-2.data")

# %%

plt.plot(mesa.star_age - mesa.star_age[-1] + mesa.star_age[-1], mesa.R)
plt.plot(nugrid.star_age - nugrid.star_age[-1] + mesa.star_age[-1], nugrid.R)

plt.xlim(0)
plt.show()
# %%
nugrid.bulk_names

# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(mesa.star_age - mesa.star_age[-1], mesa.he_core_mass, c="C0", label="Rees")
plt.plot(
    mesa_eagb.star_age - mesa_eagb.star_age[-1] - mesa.star_age[-1],
    mesa_eagb.he_core_mass,
    c="C0",
)
plt.plot(
    nugrid.star_age - nugrid.star_age[-1],
    nugrid.h1_boundary_mass,
    c="C1",
    label="NuGrid",
)


plt.xlim(-3e6, 0.1e6)
plt.ylim(0.5, 0.65)


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("time - final age (yr)")
plt.ylabel("He-core mass ($M_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w21-compare-rees-nugrid.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(mesa.star_age - mesa.star_age[-1], mesa.log_abs_mdot, c="C0", label="Rees")
plt.plot(
    mesa_eagb.star_age - mesa_eagb.star_age[-1] - mesa.star_age[-1],
    mesa_eagb.log_abs_mdot,
    c="C0",
)
plt.plot(
    nugrid.star_age - nugrid.star_age[-1], nugrid.log_abs_mdot, c="C1", label="NuGrid"
)


plt.xlim(-3e6, 0.1e6)
plt.ylim(-12, -2)


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel("time - final age (yr)")
plt.ylabel(r"$\log(\dot{M} / M_\odot \textrm{yr}^{-1})$")
plt.savefig(
    "/home/koen/LaTeX-setup/plots/w21-compare-rees-nugrid-mdot.pgf", format="pgf"
)
plt.show()
plt.close()
# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

plt.plot(
    mesa.he_core_mass,
    mesa.surface_c12 / mesa.surface_o16 * 16 / 12,
    c="C0",
    label="Rees",
)
plt.plot(
    mesa_eagb.he_core_mass,
    mesa_eagb.surface_c12 / mesa_eagb.surface_o16 * 16 / 12,
    c="C0",
)
plt.plot(
    nugrid.h1_boundary_mass,
    nugrid.surface_c12 / nugrid.surface_o16 * 16 / 12,
    c="C1",
    label="NuGrid",
)

plt.xlim(0.54, 0.64)


fig.legend(loc="outside upper center", ncols=2)
plt.xlabel(r"He-core mass ($M_\odot$)")
plt.ylabel(r"C/O-ratio")
plt.savefig("/home/koen/LaTeX-setup/plots/w21-compare-CO.pgf", format="pgf")
plt.show()
plt.close()

# %%
