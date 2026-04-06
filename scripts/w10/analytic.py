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
M_final = np.linspace(2, 1.4, 100)
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
)

plt.legend()
plt.xlim(1.26, 2.01)
axs.invert_xaxis()
plt.xlabel(r"$M_\textrm{d}$ ($M_\odot$)")
plt.ylabel(r"$a_\textrm{f} / a_\textrm{i}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-analytical1.pgf", format="pgf")
plt.show()
plt.close()

# %%


def rl(a, q):
    return a * 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + np.log(1 + q ** (1 / 3)))


# %%

fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(column), constrained_layout=True
)

M_initial = 2
M_final = np.linspace(2, 1.4, 100)
a_initial = 1400

for q in [0.5, 0.75, 1, 1.25]:
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
    )

plt.xlim(1.26, 2.01)
axs.invert_xaxis()
plt.xlabel(r"$M_\textrm{d}$ ($M_\odot$)")
plt.ylabel(r"$R_\textrm{RL}$")
plt.savefig("/home/koen/LaTeX-setup/plots/w10-analytical2.pgf", format="pgf")
plt.show()
plt.close()


# %%
