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
def a_final(a_initial, M_initial, M_final, q):
    M_tot = M_initial + q * M_initial
    return (
        a_initial
        * ((M_initial * (M_tot - M_initial)) / (M_final * (M_tot - M_final))) ** 2
    )


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

M_initial = 2
M_final = np.linspace(2, 0.6, 100)
a_initial = 1400

q = 0.5
curr_q = (M_initial * q + (M_initial - M_final)) / M_final
index = np.argwhere(curr_q > 1)[0][0]
plt.plot(
    M_final[:index],
    a_final(a_initial, M_initial, M_final, q)[:index] / a_initial,
    c="C3",
    label=r"$q < 1$",
)
plt.scatter(
    M_final[index - 1],
    a_final(a_initial, M_initial, M_final, q)[index - 1] / a_initial,
    c="C2",
    zorder=1000,
    label=r"$q = 1$",
)
plt.plot(
    M_final[index - 1 :],
    a_final(a_initial, M_initial, M_final, q)[index - 1 :] / a_initial,
    c="C0",
    label=r"$q > 1$",
)
plt.text(
    M_final[-1] - 0.02,
    a_final(a_initial, M_initial, M_final, q)[-1] / a_initial,
    f"$q_\\textrm{{i}}={q:.2f}$",
    ha="left",
    va="center",
    size=8,
)

q = 0.75
curr_q = (M_initial * q + (M_initial - M_final)) / M_final
index = np.argwhere(curr_q > 1)[0][0]
plt.plot(
    M_final[:index],
    a_final(a_initial, M_initial, M_final, q)[:index] / a_initial,
    c="C3",
)
plt.scatter(
    M_final[index - 1],
    a_final(a_initial, M_initial, M_final, q)[index - 1] / a_initial,
    c="C2",
    zorder=1000,
)
plt.plot(
    M_final[index - 1 :],
    a_final(a_initial, M_initial, M_final, q)[index - 1 :] / a_initial,
    c="C0",
)
plt.text(
    M_final[-1] - 0.02,
    a_final(a_initial, M_initial, M_final, q)[-1] / a_initial,
    f"$q_\\textrm{{i}}={q:.2f}$",
    ha="left",
    va="center",
    size=8,
)

q = 1
curr_q = (M_initial * q + (M_initial - M_final)) / M_final
index = np.argwhere(curr_q > 1)[0][0]
plt.plot(
    M_final[:index],
    a_final(a_initial, M_initial, M_final, q)[:index] / a_initial,
    c="C3",
)
plt.scatter(
    M_final[index - 1],
    a_final(a_initial, M_initial, M_final, q)[index - 1] / a_initial,
    c="C2",
    zorder=1000,
)
plt.plot(
    M_final[index - 1 :],
    a_final(a_initial, M_initial, M_final, q)[index - 1 :] / a_initial,
    c="C0",
)
plt.text(
    M_final[-1] - 0.02,
    a_final(a_initial, M_initial, M_final, q)[-1] / a_initial,
    f"$q_\\textrm{{i}}={q:.2f}$",
    ha="left",
    va="center",
    size=8,
)

q = 1.25
curr_q = (M_initial * q + (M_initial - M_final)) / M_final
plt.plot(M_final, a_final(a_initial, M_initial, M_final, q) / a_initial, c="C0")
plt.text(
    M_final[-1] - 0.02,
    a_final(a_initial, M_initial, M_final, q)[-1] / a_initial,
    f"$q_\\textrm{{i}}={q:.2f}$",
    ha="left",
    va="center",
    size=8,
)

plt.legend()
plt.xlim(0.35, 2.01)
axs.invert_xaxis()
plt.xlabel(r"$M_\textrm{d}$ ($M_\odot$)")
plt.ylabel(r"$a_\textrm{f} / a_\textrm{i}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-analytical1.pgf", format="pgf")
plt.show()
plt.close()

# %%


def rl(a, q):
    return a * 0.49 * q ** (-2 / 3) / (0.6 * q ** (-2 / 3) + np.log(1 + q ** (-1 / 3)))


def get_separation(R, q):
    return (
        (0.6 * (q) ** (-2 / 3) + np.log(1 + (q) ** (-1 / 3)))
        / (0.49 * (q) ** (-2 / 3))
        * R
    )


def evolve_q(q_initial, M1_initial, M1):
    return (q_initial * M1_initial + (M1_initial - M1)) / M1


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

M_initial = 2
M_final = np.linspace(2, 0.6, 100)
a_initial = 1000

for q in [0.5, 0.75, 1, 1.25]:
    skip_dot = False
    curr_q = (M_initial * q + (M_initial - M_final)) / M_final
    print(curr_q)
    index = np.argwhere(curr_q > 1)[0][0]
    if index == 0:
        skip_dot = True
        index = 1
    plt.plot(
        M_final[:index],
        rl(a_final(a_initial, M_initial, M_final, q)[:index], curr_q[:index]),
        c="C3",
        label=r"$q < 1$",
    )
    if not skip_dot:
        plt.scatter(
            M_final[index - 1],
            rl(a_final(a_initial, M_initial, M_final, q)[index - 1], curr_q[index - 1]),
            c="C2",
            zorder=1000,
            label=r"$q = 1$",
        )
    plt.plot(
        M_final[index - 1 :],
        rl(a_final(a_initial, M_initial, M_final, q)[index - 1 :], curr_q[index - 1 :]),
        c="C0",
        label=r"$q > 1$",
    )
    plt.text(
        M_final[-1] - 0.02,
        rl(a_final(a_initial, M_initial, M_final, q)[-1], curr_q[-1]),
        f"$q_\\textrm{{i}}={q:.2f}$",
        ha="left",
        va="center",
        size=8,
    )

plt.text(1.95, 1000, r"$a_\textrm{i} = 1000 R_\odot$")
plt.xlim(0.35, 2.01)
axs.invert_xaxis()
plt.xlabel(r"$M_\textrm{d}$ ($M_\odot$)")
plt.ylabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-analytical2.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

M_initial = 2
M_final = np.linspace(2, 1.4, 100)

for q in [0.5, 0.75, 1, 1.25]:
    a_initial = get_separation(400, q)
    print(rl(a_initial, q))
    print(a_initial)
    skip_dot = False
    curr_q = (M_initial * q + (M_initial - M_final)) / M_final
    index = np.argwhere(curr_q > 1)[0][0]
    if index == 0:
        skip_dot = True
        index = 1
    plt.plot(
        M_final[:index],
        rl(a_final(a_initial, M_initial, M_final, q)[:index], curr_q[:index]),
        c="C3",
        label=r"$q < 1$",
    )
    if not skip_dot:
        plt.scatter(
            M_final[index - 1],
            rl(a_final(a_initial, M_initial, M_final, q)[index - 1], curr_q[index - 1]),
            c="C2",
            zorder=1000,
            label=r"$q = 1$",
        )
    plt.plot(
        M_final[index - 1 :],
        rl(a_final(a_initial, M_initial, M_final, q)[index - 1 :], curr_q[index - 1 :]),
        c="C0",
        label=r"$q > 1$",
    )
    plt.text(
        M_final[-1] - 0.02,
        rl(a_final(a_initial, M_initial, M_final, q)[-1], curr_q[-1]),
        f"$q_\\textrm{{i}}={q:.2f}$",
        ha="left",
        va="center",
        size=8,
    )

plt.xlim(1.28, 2.01)
axs.invert_xaxis()
plt.xlabel(r"$M_\textrm{d}$ ($M_\odot$)")
plt.ylabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-analytical3.pgf", format="pgf")
plt.show()
plt.close()


# %%


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

M_initial = 2
M_final = 1.4

R_initials = np.linspace(150, 700, 100)
for q in np.linspace(0.4, 1, 7):
    a_fs = []
    for R_i in R_initials:
        a_initial = get_separation(R_i, q)
        a_f = a_final(a_initial, M_initial, M_final, q)
        a_fs.append(a_f)
    plt.plot(R_initials, a_fs, label=f"$q={q:.1f}$")


fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-analytical4.pgf", format="pgf")
plt.show()
plt.close()
# %%
Rsun = 6.957e10
Msun = 1.3271244e26 / 6.67430e-8
G = 6.67430e-8


def get_period(a, M, q):
    return (
        np.sqrt(4 * np.pi**2 * (a * Rsun) ** 3 / (G * (1 + q) * M * (Msun))) / 3600 / 24
    )


# %%
fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

M_initial = 2
M_final = 0.6

R_initials = np.linspace(150, 500, 100)
for q in np.linspace(0.4, 1, 7):
    a_fs = []
    for R_i in R_initials:
        a_initial = get_separation(R_i, q)
        q_f = evolve_q(q, M_initial, M_final)
        a_f = a_final(a_initial, M_initial, M_final, q)
        a_fs.append(a_f)
    a_fs = np.array(a_fs)
    plt.plot(R_initials, get_period(a_fs, M_final, q_f), label=f"$q={q:.1f}$")


plt.xlim(150, 650)
plt.ylim(1800, 15000)
fig.legend(loc="outside upper center", ncols=4)
plt.xlabel(r"$R_\textrm{RL}$ ($R_\odot$)")
plt.ylabel(r"Period (days)")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-analytical4.pgf", format="pgf")
plt.show()
plt.close()
# %%
