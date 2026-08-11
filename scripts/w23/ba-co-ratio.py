import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.style import context
import sys
import pandas as pd

sys.path.insert(1, "/home/koen/LaTeX-setup/python-files/")
from plot_size import set_size

column = 312.98032
full = 483.69684
plt.style.use("default")
plt.style.use("tex rm")

# %%
# ============================================================
# input data
# ============================================================

data = [
    [
        "HD 15096",
        -0.14,
        0.29,
        0.15,
        8.67,
        0.14,
        0.00,
        8.83,
        1.14,
        1.00,
        3.13,
        "roriz2024",
    ],
    [
        "HD 37792",
        -0.55,
        0.46,
        -0.09,
        8.43,
        0.27,
        -0.28,
        8.55,
        1.45,
        0.90,
        3.03,
        "roriz2024",
    ],
    [
        "HD 141804",
        -0.41,
        0.62,
        0.21,
        8.73,
        0.16,
        -0.25,
        8.58,
        1.71,
        1.30,
        3.43,
        "roriz2024",
    ],
    [
        "HD 207585",
        -0.34,
        0.66,
        0.32,
        8.84,
        0.06,
        -0.40,
        8.55,
        1.74,
        1.40,
        3.53,
        "roriz2024",
    ],
    [
        "HD 4395",
        -0.33,
        np.nan,
        np.nan,
        8.65,
        np.nan,
        np.nan,
        8.65,
        0.56,
        0.23,
        np.nan,
        "smith1993",
    ],
    [
        "HD 11377",
        -0.05,
        np.nan,
        np.nan,
        8.71,
        np.nan,
        np.nan,
        9.06,
        0.03,
        -0.02,
        np.nan,
        "smith1993",
    ],
    [
        "HD 88446",
        -0.36,
        np.nan,
        np.nan,
        8.41,
        np.nan,
        np.nan,
        8.86,
        0.64,
        0.28,
        np.nan,
        "smith1993",
    ],
    [
        "HD 89948",
        -0.27,
        np.nan,
        np.nan,
        8.86,
        np.nan,
        np.nan,
        8.77,
        0.83,
        0.56,
        np.nan,
        "smith1993",
    ],
    [
        "HD 125079",
        -0.16,
        np.nan,
        np.nan,
        9.05,
        np.nan,
        np.nan,
        8.73,
        0.75,
        0.59,
        np.nan,
        "smith1993",
    ],
    [
        "HD 182274",
        -0.18,
        np.nan,
        np.nan,
        8.92,
        np.nan,
        np.nan,
        9.03,
        0.59,
        0.41,
        np.nan,
        "smith1993",
    ],
    [
        "HD 204613",
        -0.35,
        np.nan,
        np.nan,
        8.91,
        np.nan,
        np.nan,
        8.95,
        0.56,
        0.21,
        np.nan,
        "smith1993",
    ],
    [
        "HD 216219",
        -0.32,
        np.nan,
        np.nan,
        9.02,
        np.nan,
        np.nan,
        8.93,
        0.89,
        0.57,
        np.nan,
        "smith1993",
    ],
    [
        "HD 219116",
        -0.34,
        np.nan,
        np.nan,
        8.73,
        np.nan,
        np.nan,
        8.48,
        0.90,
        0.56,
        np.nan,
        "smith1993",
    ],
    [
        "HR 6094",
        0.04,
        -0.18,
        -0.14,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.37,
        0.41,
        np.nan,
        "portodemello1996",
    ],
    [
        "HD 50264",
        -0.34,
        0.59,
        0.25,
        8.80,
        np.nan,
        np.nan,
        np.nan,
        1.25,
        0.91,
        3.04,
        "pereira2002",
    ],
    [
        "HD 87080",
        -0.51,
        0.61,
        0.10,
        8.65,
        np.nan,
        np.nan,
        np.nan,
        1.51,
        1.00,
        3.13,
        "pereira2002",
    ],
    [
        "HD 8270",
        -0.53,
        0.71,
        0.18,
        8.70,
        np.nan,
        np.nan,
        np.nan,
        1.17,
        0.64,
        2.77,
        "pereira2005",
    ],
    [
        "HD 13551",
        -0.28,
        0.46,
        0.18,
        8.70,
        np.nan,
        np.nan,
        np.nan,
        1.38,
        1.10,
        3.23,
        "pereira2005",
    ],
    [
        "HD 22589",
        -0.16,
        0.64,
        0.48,
        9.00,
        np.nan,
        np.nan,
        np.nan,
        0.75,
        0.59,
        2.72,
        "pereira2005",
    ],
    [
        "HD 55496",
        -1.55,
        -0.35,
        -1.90,
        6.62,
        0.15,
        -1.40,
        7.43,
        0.62,
        -0.93,
        1.20,
        "pereira2018",
    ],
    [
        "BD-03.3668",
        -0.55,
        0.59,
        0.04,
        8.60,
        0.35,
        -0.20,
        8.73,
        1.64,
        1.09,
        3.22,
        "pereira2011",
    ],
    [
        "BD+68.1027",
        -0.31,
        0.11,
        -0.20,
        np.nan,
        0.07,
        -0.24,
        np.nan,
        0.62,
        0.31,
        np.nan,
        "kong2017",
    ],
    [
        "BD+80.670",
        -0.06,
        -0.13,
        -0.19,
        np.nan,
        -0.20,
        -0.26,
        np.nan,
        0.31,
        0.25,
        np.nan,
        "kong2017",
    ],
    [
        "REJ0702+129",
        0.13,
        0.16,
        0.29,
        np.nan,
        0.17,
        0.30,
        np.nan,
        0.56,
        0.69,
        np.nan,
        "kong2017",
    ],
    [
        "HD 147609",
        -0.28,
        0.38,
        0.10,
        8.53,
        0.64,
        0.36,
        9.05,
        1.40,
        1.12,
        3.30,
        "shejeelammal2020",
    ],
    [
        "HD 154276",
        -0.10,
        np.nan,
        np.nan,
        np.nan,
        0.32,
        0.22,
        8.91,
        0.22,
        0.12,
        2.30,
        "shejeelammal2020",
    ],
    [
        "HD 2946",
        -0.27,
        -0.03,
        -0.30,
        np.nan,
        0.13,
        -0.14,
        np.nan,
        0.48,
        0.21,
        np.nan,
        "liu2021",
    ],
]

columns = [
    "id",
    "[Fe/H]",
    "[C/Fe]",
    "[C/H]",
    "log eps_C",
    "[O/Fe]",
    "[O/H]",
    "log eps_O",
    "[Ba/Fe]",
    "[Ba/H]",
    "log eps_Ba",
    "ref",
]

df = pd.DataFrame(data, columns=columns)


# ============================================================
# solar abundances
# ============================================================

# Asplund et al. solar abundances.
# These are only used when log eps_C or log eps_O is unavailable.

logeps_C_sun = 8.46
logeps_O_sun = 8.69

CO_sun = 10 ** (logeps_C_sun - logeps_O_sun)


# ============================================================
# calculate C/O
# ============================================================


def calculate_CO(row):
    """
    Calculate C/O.

    Priority:
        1. log eps_C and log eps_O
        2. [C/H] and [O/H] using the adopted solar abundances
        3. NaN if neither is possible
    """

    # --------------------------------------------------------
    # preferred method: directly measured log epsilon values
    # --------------------------------------------------------
    if pd.notna(row["log eps_C"]) and pd.notna(row["log eps_O"]):
        return (
            10 ** (row["log eps_C"] - row["log eps_O"]),
            "log eps",
        )

    # --------------------------------------------------------
    # fallback: bracket abundances
    # --------------------------------------------------------
    if pd.notna(row["[C/H]"]) and pd.notna(row["[O/H]"]):
        return (
            CO_sun * 10 ** (row["[C/H]"] - row["[O/H]"]),
            "[C/H], [O/H]",
        )

    # --------------------------------------------------------
    # insufficient information
    # --------------------------------------------------------
    return np.nan, None


df[["C/O", "C/O method"]] = df.apply(
    calculate_CO,
    axis=1,
    result_type="expand",
)


# ============================================================
# inspect the resulting sample
# ============================================================

plot_df = df.dropna(subset=["C/O"]).copy()

print(plot_df[["id", "[Fe/H]", "C/O", "C/O method", "ref"]].to_string(index=False))

print()
print(f"total stars in table: {len(df)}")
print(f"stars with C/O:       {len(plot_df)}")
print(f"  using log eps:      " f"{(plot_df['C/O method'] == 'log eps').sum()}")
print(f"  using [C/H],[O/H]:  " f"{(plot_df['C/O method'] == '[C/H], [O/H]').sum()}")


# ============================================================
# plot
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

# use different markers depending on whether we had direct
# log-epsilon abundances or had to reconstruct C/O
direct = plot_df["C/O method"] == "log eps"
fallback = plot_df["C/O method"] == "[C/H], [O/H]"

ax.scatter(
    plot_df.loc[direct, "[Fe/H]"],
    plot_df.loc[direct, "C/O"],
    s=60,
    label=r"direct $\log\epsilon$",
)

ax.scatter(
    plot_df.loc[fallback, "[Fe/H]"],
    plot_df.loc[fallback, "C/O"],
    s=60,
    marker="x",
    label=r"from [C/H], [O/H]",
)

# carbon-rich / oxygen-rich boundary
ax.axhline(
    1,
    linestyle="--",
    linewidth=1,
)

# annotate stars
for _, row in plot_df.iterrows():
    ax.annotate(
        row["ref"],
        (row["[Fe/H]"], row["C/O"]),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=7,
    )

ax.set_xlabel(r"$[\mathrm{Fe/H}]$")
ax.set_ylabel(r"$\mathrm{C/O}$")
ax.set_title("barium stars: metallicity vs. C/O")

ax.grid(
    alpha=0.25,
)

ax.legend()

fig.tight_layout()
plt.show()
# %%
# ============================================================
# solar abundances
# ============================================================

# Asplund et al. solar abundances.
# These are only used when log eps_C or log eps_O is unavailable.

logeps_C_sun = 8.46
logeps_O_sun = 8.69

CO_sun = 10 ** (logeps_C_sun - logeps_O_sun)


# ============================================================
# calculate C/O
# ============================================================


def calculate_CO(row):
    """
    Calculate C/O.

    Priority:
        1. log eps_C and log eps_O
        2. [C/H] and [O/H] using the adopted solar abundances
        3. NaN if neither is possible
    """

    # --------------------------------------------------------
    # preferred method: directly measured log epsilon values
    # --------------------------------------------------------
    if pd.notna(row["log eps_C"]) and pd.notna(row["log eps_O"]):
        return (
            10 ** (row["log eps_C"] - row["log eps_O"]),
            "log eps",
        )

    # --------------------------------------------------------
    # fallback: bracket abundances
    # --------------------------------------------------------
    if pd.notna(row["[C/H]"]) and pd.notna(row["[O/H]"]):
        return (
            CO_sun * 10 ** (row["[C/H]"] - row["[O/H]"]),
            "[C/H], [O/H]",
        )

    # --------------------------------------------------------
    # insufficient information
    # --------------------------------------------------------
    return np.nan, None


df[["C/O", "C/O method"]] = df.apply(
    calculate_CO,
    axis=1,
    result_type="expand",
)


# ============================================================
# inspect the resulting sample
# ============================================================

plot_df = df.dropna(subset=["C/O"]).copy()

print(plot_df[["id", "[Fe/H]", "C/O", "C/O method", "ref"]].to_string(index=False))

print()
print(f"total stars in table: {len(df)}")
print(f"stars with C/O:       {len(plot_df)}")
print(f"  using log eps:      " f"{(plot_df['C/O method'] == 'log eps').sum()}")
print(f"  using [C/H],[O/H]:  " f"{(plot_df['C/O method'] == '[C/H], [O/H]').sum()}")


# ============================================================
# plot
# ============================================================

fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


# use different markers depending on whether we had direct
# log-epsilon abundances or had to reconstruct C/O
direct = plot_df["C/O method"] == "log eps"
fallback = plot_df["C/O method"] == "[C/H], [O/H]"

for ref, group in plot_df.groupby("ref"):
    if len(group.loc[direct, "[Ba/Fe]"]) != 0:
        ax.scatter(
            group.loc[direct, "[Ba/Fe]"], group.loc[direct, "C/O"], s=15, label=ref
        )

    if len(group.loc[fallback, "[Ba/Fe]"]) != 0:
        ax.scatter(
            group.loc[fallback, "[Ba/Fe]"],
            group.loc[fallback, "C/O"],
            s=15,
            marker="s",
            label=ref,
        )

# carbon-rich / oxygen-rich boundary
ax.axhline(1, linewidth=0.8, c="C9", zorder=-1)

ax.set_xlabel(r"$[\mathrm{Ba/Fe}]$")
ax.set_ylabel(r"C/O-ratio")
#
# ax.grid(
#     alpha=0.25,
# )

fig.legend(loc="outside upper center", ncols=4, handlelength=1)


ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w23-co-ratio.pgf", format="pgf")
plt.show()
plt.close()
# %%

# ============================================================
# solar abundances
# ============================================================

# Asplund et al. solar abundances.
# These are only used when log eps_C or log eps_O is unavailable.

logeps_C_sun = 8.46
logeps_O_sun = 8.69

CO_sun = 10 ** (logeps_C_sun - logeps_O_sun)


# ============================================================
# calculate C/O
# ============================================================


def calculate_CO(row):
    """
    Calculate C/O.

    Priority:
        1. log eps_C and log eps_O
        2. [C/H] and [O/H] using the adopted solar abundances
        3. NaN if neither is possible
    """

    # --------------------------------------------------------
    # preferred method: directly measured log epsilon values
    # --------------------------------------------------------
    if pd.notna(row["log eps_C"]) and pd.notna(row["log eps_O"]):
        return (
            10 ** (row["log eps_C"] - row["log eps_O"]),
            "log eps",
        )

    # --------------------------------------------------------
    # fallback: bracket abundances
    # --------------------------------------------------------
    if pd.notna(row["[C/H]"]) and pd.notna(row["[O/H]"]):
        return (
            CO_sun * 10 ** (row["[C/H]"] - row["[O/H]"]),
            "[C/H], [O/H]",
        )

    # --------------------------------------------------------
    # insufficient information
    # --------------------------------------------------------
    return np.nan, None


df[["C/O", "C/O method"]] = df.apply(
    calculate_CO,
    axis=1,
    result_type="expand",
)


# ============================================================
# inspect the resulting sample
# ============================================================

plot_df = df.dropna(subset=["C/O"]).copy()

print(plot_df[["id", "[Fe/H]", "C/O", "C/O method", "ref"]].to_string(index=False))

print()
print(f"total stars in table: {len(df)}")
print(f"stars with C/O:       {len(plot_df)}")
print(f"  using log eps:      " f"{(plot_df['C/O method'] == 'log eps').sum()}")
print(f"  using [C/H],[O/H]:  " f"{(plot_df['C/O method'] == '[C/H], [O/H]').sum()}")


# ============================================================
# plot
# ============================================================

fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)


# use different markers depending on whether we had direct
# log-epsilon abundances or had to reconstruct C/O
direct = plot_df["C/O method"] == "log eps"
fallback = plot_df["C/O method"] == "[C/H], [O/H]"

for ref, group in plot_df.groupby("ref"):
    if len(group.loc[direct, "[Fe/H]"]) != 0:
        ax.scatter(
            group.loc[direct, "[Fe/H]"], group.loc[direct, "C/O"], s=15, label=ref
        )

    if len(group.loc[fallback, "[Fe/H]"]) != 0:
        ax.scatter(
            group.loc[fallback, "[Fe/H]"],
            group.loc[fallback, "C/O"],
            s=15,
            marker="s",
            label=ref,
        )

# carbon-rich / oxygen-rich boundary
ax.axhline(1, linewidth=0.8, c="C9", zorder=-1)

ax.set_xlabel(r"$[\mathrm{Fe/H}]$")
ax.set_ylabel(r"C/O-ratio")
#
# ax.grid(
#     alpha=0.25,
# )

fig.legend(loc="outside upper center", ncols=4, handlelength=1)


ax.spines[["right", "top"]].set_visible(False)
plt.savefig("/home/koen/LaTeX-setup/plots/w23-co-ratio-metallicity.pgf", format="pgf")
plt.show()
plt.close()
# %%
fig, ax = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

data = pd.read_csv("presentation-1/Ba_star_orbits.csv")
dwarf = data[data["class"].isin(["dBa", "?dBa"])]
giant = data[
    data["class"].isin(
        [
            "Ba 0",
            "Ba 0.5",
            "Ba 1",
            "Ba 2",
            "Ba mild",
            "sgCH",
            "sgCH / Ba 1",
            "Ba 3",
            "Ba 4",
            "Ba 5",
            "Ba strong",
            "Ba 5 / eS",
        ]
    )
]

dwarf1 = dwarf[pd.notna(dwarf["[Ba/Fe]"])]
dwarf2 = dwarf1[pd.notna(dwarf1["[Fe/H]"])]

plt.scatter(
    dwarf1["[Fe/H]"],
    dwarf1["[Ba/Fe]"],
    s=15,
    c="C0",
    label=r"\texttt{Ba\_star\_orbits.csv}",
)


plt.scatter(
    plot_df["[Fe/H]"],
    plot_df["[Ba/Fe]"],
    s=15,
    c="C1",
    label=r"New sample",
)


#
# for ref, group in plot_df.groupby("ref"):
#     if len(group.loc[direct, "[Fe/H]"]) != 0:
#         ax.scatter(
#             group.loc[direct, "[Fe/H]"],
#             group.loc[direct, "[Ba/Fe]"],
#             s=30,
#             label=ref,
#             marker="x",
#             c="C9",
#         )
#
#     if len(group.loc[fallback, "[Ba/Fe]"]) != 0:
#         ax.scatter(
#             group.loc[fallback, "[Fe/H]"],
#             group.loc[fallback, "[Ba/Fe]"],
#             s=30,
#             marker="x",
#             label=ref,
#             c="C9",
#         )

fig.legend(loc="outside upper center", ncols=2)


ax.spines[["right", "top"]].set_visible(False)
plt.xlabel("[Fe/H]")
plt.ylabel("[Ba/Fe]")
plt.savefig("/home/koen/LaTeX-setup/plots/w23-new-old-ba-fe.pgf", format="pgf")
plt.show()
plt.close()

# %%
