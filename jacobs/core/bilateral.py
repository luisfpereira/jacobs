"""Plots comparing left and right structures side by side."""

from jacobs.core.style import COLORS

SIDE_COLORS = {"Left": COLORS["left"], "Right": COLORS["right"]}
SIDE_OFFSETS = {"Left": -0.2, "Right": 0.2}
MARKER_STYLE = {"color": "black", "linestyle": "--", "linewidth": 1.0}


def group_by_side(items):
    """Regroup {"Left X": value, "Right X": value} into {"X": {"Left": value, "Right": value}}."""
    pairs = {}
    for struct, value in items.items():
        side, name = struct.split(" ", 1)
        pairs.setdefault(name, {})[side] = value
    return pairs


def plot_boxes(ax, positions, values, color, side):
    """Draw one box per position, labelled with the side."""
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for box in boxes["boxes"]:
        box.set_facecolor(color)
    boxes["boxes"][0].set_label(side)


def plot_bars(ax, positions, values, color, side):
    """Draw one bar per position, labelled with the side."""
    ax.bar(positions, values, width=0.4, color=color, edgecolor="black", label=side)


def plot_bilateral(ax, data, plot_side, tick_labels=None, reference=None):
    """Plot left and right values of each structure side by side.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to plot.
    data : dict
        Values keyed by long structure name (e.g. ``"Left Amygdala"``).
    plot_side : callable
        Called as ``plot_side(ax, positions, values, color, side)`` for each
        side, e.g. ``plot_boxes`` or ``plot_bars``.
    tick_labels : dict or None
        Tick labels keyed by structure name without side. Structures without a
        label use their name.
    reference : float or None
        If given, value of a dotted horizontal reference line.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    pairs = group_by_side(data)
    tick_labels = {} if tick_labels is None else tick_labels

    for side in ("Left", "Right"):
        plot_side(
            ax,
            [i + SIDE_OFFSETS[side] for i in range(len(pairs))],
            [pair[side] for pair in pairs.values()],
            SIDE_COLORS[side],
            side,
        )

    if reference is not None:
        ax.axhline(reference, color="gray", linestyle=":")

    ax.set_xticks(
        range(len(pairs)),
        [tick_labels.get(name, name) for name in pairs],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    return ax


def add_side_markers(ax, values, names, width=0.3, label=None):
    """Draw a short dashed line at each structure's left and right position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes drawn with ``plot_bilateral``.
    values : dict
        Marker values keyed by long structure name (e.g. ``"Left Amygdala"``).
    names : sequence of str
        Structure names without side, in the order used by ``plot_bilateral``.
    width : float
        Length of each line, in position units.
    label : str or None
        Legend label, given to the first line only.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    pairs = group_by_side(values)

    for i, name in enumerate(names):
        for side, offset in SIDE_OFFSETS.items():
            position = i + offset
            ax.hlines(
                pairs[name][side],
                position - width / 2,
                position + width / 2,
                label=label,
                **MARKER_STYLE,
            )
            label = None

    return ax
