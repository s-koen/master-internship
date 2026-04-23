import numpy as np
import matplotlib.pyplot as plt

# %%

G = 1  # normalized


def roche_potential(x, y, a=1, M1=2, q=0.5):
    """Computes the potential of a particle in the plane of the binary at position (x, y) in the rotating frame of a binary with separation a, mass ratio q and primary mass M1. The stars are assumed to lie on the x-axis with their centre of mass at (0, 0).

    The potential is then given by:
    Phi(x,y) = - GM1 / r1 - GM2 / r2 - 1/2 * Omega^2 (x^2 + y^2)

    The function returns the potential of the particle."""

    M2 = q * M1
    Mtot = M1 + M2

    x1 = -M2 / Mtot * a  # position of the primary star
    x2 = +M1 / Mtot * a  # position of the primary star

    r1 = np.sqrt((x - x1) ** 2 + y**2)
    r2 = np.sqrt((x - x2) ** 2 + y**2)

    Omega2 = G * Mtot / (a**3)  # angular velocity

    Phi_grav_1 = -G * M1 / r1
    Phi_grav_2 = -G * M2 / r2
    Phi_centri = -1 / 2 * Omega2 * (x**2 + y**2)

    return Phi_grav_1 + Phi_grav_2 + Phi_centri


def grad_roche(x, y, a=1, M1=2, q=0.5):
    M2 = q * M1
    Mtot = M1 + M2

    x1 = -M2 / (M1 + M2) * a
    x2 = M1 / (M1 + M2) * a

    r1 = np.sqrt((x - x1) ** 2 + y**2)
    r2 = np.sqrt((x - x2) ** 2 + y**2)

    Omega2 = G * (M1 + M2) / a**3

    dPhidx = G * M1 * (x - x1) / r1**3 + G * M2 * (x - x2) / r2**3 - Omega2 * x

    dPhidy = G * M1 * y / r1**3 + G * M2 * y / r2**3 - Omega2 * y

    return np.array([dPhidx, dPhidy])


# %%

# grid
x = np.linspace(-1, 1, 500)
y = np.linspace(-1, 1, 500)
X, Y = np.meshgrid(x, y)

M1 = 2 * 0.2
q = 1 / 0.2
a = 1

Phi = roche_potential(X, Y, a=a, M1=M1, q=q)

plt.figure(figsize=(6, 6))
levels = np.linspace(0.5, 1.5, 20)
print(levels)

plt.contourf(X, Y, np.log10(-Phi), levels=levels, cmap="Blues", extend="both")

# plot masses
M2 = q * M1
Mtot = M1 + M2

x1 = -M2 / Mtot * a  # position of the primary star
x2 = +M1 / Mtot * a  # position of the primary star


x1 = -M2 / (M1 + M2) * a
x2 = M1 / (M1 + M2) * a
# plt.plot([x1, x2], [0, 0], "ro")

plt.xlim(-0.85, 0.2)
plt.ylim(-0.2, 0.85)
plt.gca().set_aspect("equal")
plt.savefig("roche_potential.svg", format="svg", bbox_inches="tight")
plt.show()
# %%
