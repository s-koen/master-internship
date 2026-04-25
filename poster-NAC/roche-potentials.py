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
from scipy.optimize import root


def find_lagrange(initial_guess, a, M1, q):
    def f(vec):
        return grad_roche(vec[0], vec[1], a, M1, q)

    sol = root(f, initial_guess)
    return sol.x


M1 = 1 * 0.2
q = 1 / 0.2
a = 1

L1 = find_lagrange([0.0, 0.0], a, M1, q)
L2 = find_lagrange([1.2, 0.0], a, M1, q)
L3 = find_lagrange([-1.2, 0.0], a, M1, q)

# triangular points
L4 = find_lagrange([0.5, 0.8], a, M1, q)
L5 = find_lagrange([0.5, -0.8], a, M1, q)

print("L1:", L1)
print("L2:", L2)
print("L3:", L3)
print("L4:", L4)
print("L5:", L5)

# %%
# grid
x = np.linspace(-2, 2, 500)
y = np.linspace(-1, 1, 500)
X, Y = np.meshgrid(x, y)

M1 = 2 * 0.2
q = 1 / 0.2
a = 1

Phi = roche_potential(X, Y, a=a, M1=M1, q=q)

plt.figure(figsize=(6, 6))


# plot masses
M2 = q * M1
Mtot = M1 + M2

x1 = -M2 / Mtot * a  # position of the primary star
x2 = +M1 / Mtot * a  # position of the primary star


x1 = -M2 / (M1 + M2) * a
x2 = M1 / (M1 + M2) * a
# plt.plot([x1, x2], [0, 0], "ro")

for L in [L1, L2, L3, L4, L5]:
    plt.plot(L[0], L[1], "kx")

Phi_L1 = np.log10(-roche_potential(L1[0], L1[1], a, M1, q))
Phi_L5 = np.log10(-roche_potential(L5[0], L5[1], a, M1, q))

levels = np.logspace(-0.3, 0.3, 21)
print(levels)
plt.contourf(
    X,
    Y,
    np.log10(-Phi),
    levels=Phi_L1 * levels,
    cmap="Blues",
    extend="both",
    vmin=Phi_L5,
)
plt.contour(
    X,
    Y,
    np.log10(-Phi),
    levels=Phi_L1 * levels,
    cmap="Blues",
    extend="both",
    vmin=Phi_L5,
)

plt.gca().set_aspect("equal")
# plt.savefig("roche_potential.svg", format="svg", bbox_inches="tight")
plt.show()
# %%

x = np.linspace(-2, 2, 500)
y = np.linspace(-2, 2, 500)
X, Y = np.meshgrid(x, y)

M1 = 2 * 0.2
q = 1 / 0.2
a = 1

Phi = roche_potential(X, Y, a=a, M1=M1, q=q)

L1 = find_lagrange([0.0, 0.0], a, M1, q)
L2 = find_lagrange([1.2, 0.0], a, M1, q)
L3 = find_lagrange([-1.2, 0.0], a, M1, q)

# triangular points
L4 = find_lagrange([0.5, 0.8], a, M1, q)
L5 = find_lagrange([0.5, -0.8], a, M1, q)

print("L1:", L1)
print("L2:", L2)
print("L3:", L3)
print("L4:", L4)
print("L5:", L5)


plt.figure(figsize=(6, 6))


Phi_L1 = np.log10(-roche_potential(L1[0], L1[1], a, M1, q))
Phi_L5 = np.log10(-roche_potential(L5[0], L5[1], a, M1, q))

plt.contour(
    X, Y, np.log10(-Phi), levels=[Phi_L1], cmap="Blues", extend="both", vmin=Phi_L5
)

plt.gca().set_aspect("equal")
plt.savefig("roche_lobe.svg", format="svg", bbox_inches="tight")
plt.show()
# %%
