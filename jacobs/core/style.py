import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from polpo.plot.pyplot import figsize_from_textwidth

from .study import resolve_subject_ids

COLORS = {
    "original": "#DE815A",
    "reconstructed": "#885FA2",
    "global": "#F8E081",
    "within": "#D995A1",
    "between": "#4D4DC1",
}


# A4 with 1in margins, in LaTeX points
TEXTWIDTH = 453.0
# subfigure width as a fraction of the text width, shared by panels and documents
SUBFIG_FRACTION = 0.48

PAPER_RC = {
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "axes.titlesize": 10,
    "axes.titlepad": 10,
    # resolution of rasterized artists (e.g. distance scatters)
    "savefig.dpi": 200,
    # no tight bbox: the saved size is exactly figsize, so panels print unscaled
    "savefig.bbox": "standard",
}

PAPER_GRID = {
    # two rows of panels must fit the text height (~700pt)
    "figsize": figsize_from_textwidth(TEXTWIDTH, SUBFIG_FRACTION, aspect=1.55),
    # one legend entry per row, so the legend fits the panel width
    "legend_kwargs": {"ncol": 1},
    # tight padding around the shared legend, explicit space between rows
    "layout": {"h_pad": 0.04, "hspace": 0.15},
}

PAPER_SUBFIGURES = {"width": SUBFIG_FRACTION, "n_cols": 2}

PAPER_DOCUMENT = {"paper": "a4paper"}


def update_mpl_params(format="pdf"):
    mpl.rcParams.update(
        {
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "savefig.format": format,
            "image.cmap": "bwr",
            "pdf.fonttype": 42,
        }
    )


def get_subject_colors(subj_ids=None):
    subj_ids = resolve_subject_ids(subj_ids)

    color_ids = [
        subj_id for subj_id in subj_ids if not str(subj_id).startswith(("3", "4"))
    ]

    tab10 = plt.colormaps["tab10"]
    base_colors = [tab10(i) for i in range(tab10.N) if i != 7]

    cmap = mcolors.ListedColormap(base_colors).resampled(len(color_ids))

    colors = dict(zip(color_ids, cmap(range(len(color_ids)))))
    colors.update(
        {subj_id: "gray" for subj_id in subj_ids if str(subj_id).startswith(("3", "4"))}
    )
    return colors
