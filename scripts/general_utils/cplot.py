import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def cplot(
    x,
    y,
    c,
    *,
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    rasterized=True,
    **kwargs,
):
    """
    Plot a line whose segments are colored according to the average
    value of `c` at the endpoints of each segment.

    Parameters
    ----------
    x, y, c : array-like
        Arrays of equal length containing the x positions, y positions,
        and color values.

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used to map `c` values to colors.

    norm : matplotlib.colors.Normalize, optional
        Normalization used for mapping `c` values to the colormap.
        If None, matplotlib automatically uses the min/max of `c`.

    rasterized : bool, optional
        Whether to rasterize the line collection when rendering.
        Defaults to True, which is useful for large datasets in PDFs.

    **kwargs
        Additional arguments passed to matplotlib's LineCollection.
        This includes linewidth, linestyle, alpha, zorder, label, etc.

    Returns
    -------
    matplotlib.collections.LineCollection
        The created line collection.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)

    if not (len(x) == len(y) == len(c)):
        raise ValueError("x, y, and c must have the same length")

    if len(x) < 2:
        raise ValueError("x, y, and c must contain at least two points")

    if norm is None:
        norm = plt.Normalize(
            vmin=np.min(c) if vmin is None else vmin,
            vmax=np.max(c) if vmax is None else vmax,
        )

    points = np.column_stack((x, y))
    segments = np.stack((points[:-1], points[1:]), axis=1)

    c_segments = 0.5 * (c[:-1] + c[1:])

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        rasterized=rasterized,
        **kwargs,
    )

    lc.set_array(c_segments)

    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale_view()

    return lc
