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
def streamQuiver(ax, sp, *args, spacing=None, n=5, **kwargs):
    """Plot arrows from streamplot data
    The number of arrows per streamline is controlled either by `spacing` or by `n`.
    See `lines_to_arrows`.
    """

    def curve_coord(line=None):
        """return curvilinear coordinate"""
        x = line[:, 0]
        y = line[:, 1]
        s = np.zeros(x.shape)
        s[1:] = np.sqrt((x[1:] - x[0:-1]) ** 2 + (y[1:] - y[0:-1]) ** 2)
        s = np.cumsum(s)
        return s

    def curve_extract(line, spacing, offset=None):
        """Extract points at equidistant space along a curve"""
        x = line[:, 0]
        y = line[:, 1]
        if offset is None:
            offset = spacing / 2
        # Computing curvilinear length
        s = curve_coord(line)
        offset = np.mod(offset, s[-1])  # making sure we always get one point
        # New (equidistant) curvilinear coordinate
        sExtract = np.arange(offset, s[-1], spacing)
        # Interpolating based on new curvilinear coordinate
        xx = np.interp(sExtract, s, x)
        yy = np.interp(sExtract, s, y)
        return np.array([xx, yy]).T

    def seg_to_lines(seg):
        """Convert a list of segments to a list of lines"""

        def extract_continuous(i):
            x = []
            y = []
            # Special case, we have only 1 segment remaining:
            if i == len(seg) - 1:
                x.append(seg[i][0, 0])
                y.append(seg[i][0, 1])
                x.append(seg[i][1, 0])
                y.append(seg[i][1, 1])
                return i, x, y
            # Looping on continuous segment
            while i < len(seg) - 1:
                # Adding our start point
                x.append(seg[i][0, 0])
                y.append(seg[i][0, 1])
                # Checking whether next segment continues our line
                Continuous = all(seg[i][1, :] == seg[i + 1][0, :])
                if not Continuous:
                    # We add our end point then
                    x.append(seg[i][1, 0])
                    y.append(seg[i][1, 1])
                    break
                elif i == len(seg) - 2:
                    # we add the last segment
                    x.append(seg[i + 1][0, 0])
                    y.append(seg[i + 1][0, 1])
                    x.append(seg[i + 1][1, 0])
                    y.append(seg[i + 1][1, 1])
                i = i + 1
            return i, x, y

        lines = []
        i = 0
        while i < len(seg):
            iEnd, x, y = extract_continuous(i)
            lines.append(np.array([x, y]).T)
            i = iEnd + 1
        return lines

    def lines_to_arrows(lines, n=5, spacing=None, normalize=True):
        """Extract "streamlines" arrows from a set of lines
        Either: `n` arrows per line
            or an arrow every `spacing` distance
        If `normalize` is true, the arrows have a unit length
        """
        if spacing is None:
            # if n is provided we estimate the spacing based on each curve lenght)
            spacing = [curve_coord(l)[-1] / n for l in lines]
        try:
            len(spacing)
        except:
            spacing = [spacing] * len(lines)

        lines_s = [
            curve_extract(l, spacing=sp, offset=sp / 2) for l, sp in zip(lines, spacing)
        ]
        lines_e = [
            curve_extract(l, spacing=sp, offset=sp / 2 + 0.01 * sp)
            for l, sp in zip(lines, spacing)
        ]
        arrow_x = [l[i, 0] for l in lines_s for i in range(len(l))]
        arrow_y = [l[i, 1] for l in lines_s for i in range(len(l))]
        arrow_dx = [
            le[i, 0] - ls[i, 0]
            for ls, le in zip(lines_s, lines_e)
            for i in range(len(ls))
        ]
        arrow_dy = [
            le[i, 1] - ls[i, 1]
            for ls, le in zip(lines_s, lines_e)
            for i in range(len(ls))
        ]

        if normalize:
            dn = [np.sqrt(ddx**2 + ddy**2) for ddx, ddy in zip(arrow_dx, arrow_dy)]
            arrow_dx = [ddx / ddn for ddx, ddn in zip(arrow_dx, dn)]
            arrow_dy = [ddy / ddn for ddy, ddn in zip(arrow_dy, dn)]
        return arrow_x, arrow_y, arrow_dx, arrow_dy

    # --- Main body of streamQuiver
    # Extracting lines
    seg = sp.lines.get_segments()  # list of (2, 2) numpy arrays
    lines = seg_to_lines(seg)  # list of (N,2) numpy arrays
    # Convert lines to arrows
    ar_x, ar_y, ar_dx, ar_dy = lines_to_arrows(
        lines, spacing=spacing, n=n, normalize=True
    )
    # Plot arrows
    qv = ax.quiver(ar_x, ar_y, ar_dx, ar_dy, *args, angles="xy", **kwargs)
    return qv


# %%


x = np.linspace(-3, 3, 500)
y = np.linspace(-2, 2, 500)
X, Y = np.meshgrid(x, y)

M1 = 2 * 0.2
q = 1 / 0.2
a = 1

Mtot = M1 + q * M1

x1 = -M2 / Mtot * a  # position of the primary star
x2 = +M1 / Mtot * a  # position of the primary star


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


fig, axs = plt.subplots(
    1, 1, sharex=True, figsize=set_size(full), constrained_layout=True
)

Phi_L1 = np.log10(-roche_potential(L1[0], L1[1], a, M1, q))
Phi_L2 = np.log10(-roche_potential(L2[0], L2[1], a, M1, q))
Phi_L3 = np.log10(-roche_potential(L3[0], L3[1], a, M1, q))
Phi_L4 = np.log10(-roche_potential(L4[0], L4[1], a, M1, q))
Phi_L5 = np.log10(-roche_potential(L5[0], L5[1], a, M1, q)) - 1e-10

print(Phi_L5, Phi_L2, Phi_L3, Phi_L1)

cs = plt.contour(
    X,
    Y,
    np.log10(-Phi),
    levels=[Phi_L5, Phi_L2, Phi_L3, Phi_L1],
    extend="both",
    linewidths=1,
    rasterized=True,
)
cs.set_rasterized(True)


cs = plt.contourf(
    X,
    Y,
    np.log10(-Phi),
    levels=[Phi_L5, Phi_L2, Phi_L3, Phi_L1],
    alpha=0.1,
)
cs.set_rasterized(True)


for L in [L1, L2, L3, L4, L5]:
    plt.scatter(L[0], L[1], color="k", zorder=10)


# coarse grid for arrows
xq = np.linspace(-3, 3, 1500)
yq = np.linspace(-2, 2, 1500)
Xq, Yq = np.meshgrid(xq, yq)

U = np.zeros_like(Xq)
V = np.zeros_like(Yq)

for i in range(Xq.shape[0]):
    for j in range(Xq.shape[1]):
        grad = grad_roche(Xq[i, j], Yq[i, j], M1, M2, a)

        # force = -grad(phi)
        U[i, j] = -grad[0]
        V[i, j] = -grad[1]

mag = np.sqrt(U**2 + V**2)
U /= mag + 1e-8
V /= mag + 1e-8


R = 1.5
mask = X**2 + Y**2 > R**2

U[mask] = np.nan
V[mask] = np.nan
dPhidy, dPhidx = np.gradient(Phi, y, x)

speed = np.log10(np.sqrt(dPhidy**2 + dPhidx**2))
speed += 1
speed = np.clip(speed, 1e-9, 2)
speed[speed == 2] = 1e-9
speed *= 2

lw = speed / np.log10(np.max(np.sqrt(dPhidy**2 + dPhidx**2)))
stream = axs.streamplot(
    X,
    Y,
    -dPhidx,
    -dPhidy,
    arrowsize=0,
    color="k",
    density=1,
    linewidth=lw,
    broken_streamlines=False,
)
stream.lines.set_rasterized(True)
stream.arrows.set_rasterized(True)

segments = stream.lines.get_segments()
colors = []

# star positions
x1_pos = -M2 / (M1 + M2) * a
x2_pos = M1 / (M1 + M2) * a

values = []

x1_pos = -M2 / (M1 + M2) * a
x2_pos = M1 / (M1 + M2) * a

for seg in segments:
    (x0, y0), (x1, y1) = seg

    r1_start = np.sqrt((x0 - x1_pos) ** 2 + y0**2)
    r2_start = np.sqrt((x0 - x2_pos) ** 2 + y0**2)

    r1_end = np.sqrt((x1 - x1_pos) ** 2 + y1**2)
    r2_end = np.sqrt((x1 - x2_pos) ** 2 + y1**2)

    d_start = min(r1_start, r2_start)
    d_end = min(r1_end, r2_end)

    values.append(d_end - d_start)

values = np.array(values)

vmax = np.max(np.abs(values))
normed = values / (vmax + 1e-12)
import matplotlib.cm as cm

cmap = cm.get_cmap("coolwarm")  # nice diverging map
colors = cmap(0.5 * (normed + 1.0))
# maps [-1,1] → [0,1]
stream.lines.set_color(colors)

stream.arrows.set(visible=False, alpha=0)

plt.savefig("/home/koen/LaTeX-setup/plots/w14-roche.pgf", format="pgf", dpi=600)
plt.show()
plt.show()
# %%
