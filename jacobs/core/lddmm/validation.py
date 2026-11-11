"""Validation figures of LDDMM-to-global outputs."""

import matplotlib as mpl
import numpy as np
from polpo.dataset.plot import plot_nested
from polpo.distmat.plot import plot_dist_comparison, plot_distmat
from polpo.latex import add_subfigures, compile_latex, make_figure_document
from polpo.plot.pyplot import plot_grid, plot_hists, plot_layers
from polpo.workflow.task import TaskRunner, task

from jacobs.core.style import (
    COLORS,
    PAPER_DOCUMENT,
    PAPER_GRID,
    PAPER_RC,
    PAPER_SUBFIGURES,
)


def is_intra(key_a, key_b):
    """Check whether two sample keys belong to the same subject.

    Parameters
    ----------
    key_a, key_b : tuple
        Sample keys, with the subject id as second entry.

    Returns
    -------
    bool
    """
    return key_a[1] == key_b[1]


def get_group_labels(metric_name):
    """Legend labels of within- and between-subject distance pairs.

    Parameters
    ----------
    metric_name : str
        Display name of the metric (e.g. ``"varifold"``).

    Returns
    -------
    within : str
    between : str
    """
    prefix = f"Global-local pairwise {metric_name} distances"
    return f"{prefix} within subjects", f"{prefix} between subjects"


HIST_KWARGS = {
    "density": True,
    "edgecolor": "black",
    # thin outlines, so narrow bars stay visible
    "linewidth": 0.3,
    "alpha": 0.65,
}


def _get_volume(point):
    """Volume enclosed by a mesh point."""
    return point.as_surface().to_polydata().volume


class BaseFiguresRunner(TaskRunner):
    """Base task runner for figure generation.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory where figures are saved.
    state_dir : pathlib.Path or None
        Directory storing the task state. If None, ``figures_dir``.
    prefix : str
        Prefix of prefixed figure names.
    sep : str
        Separator between prefix and figure name.
    """

    def __init__(self, figures_dir, state_dir=None, prefix="", sep="_"):
        if state_dir is None:
            state_dir = figures_dir

        super().__init__(state_dir)

        self.figures_dir = figures_dir
        self.prefix = prefix
        self.sep = sep

    def _get_figpath(self, name, prefixed=False):
        """Path of a PDF figure, optionally prefixed."""
        prefix = f"{self.prefix}{self.sep}" if prefixed else ""
        return self.figures_dir / f"{prefix}{name}.pdf"


class GridFiguresRunner(BaseFiguresRunner):
    """Base task runner for figures laid out with ``plot_grid``.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory where figures are saved.
    state_dir : pathlib.Path or None
        Directory storing the task state. If None, ``figures_dir``.
    prefix : str
        Prefix of prefixed figure names.
    sep : str
        Separator between prefix and figure name.
    rc : dict or None
        Matplotlib rc parameters applied while plotting and saving.
        If None, ``PAPER_RC``.
    grid : dict or None
        Default keyword arguments of ``plot_grid``. If None, ``PAPER_GRID``.
    """

    def __init__(
        self, figures_dir, state_dir=None, prefix="", sep="_", rc=None, grid=None
    ):
        super().__init__(figures_dir, state_dir=state_dir, prefix=prefix, sep=sep)

        self.rc = PAPER_RC if rc is None else rc
        self.grid = PAPER_GRID if grid is None else grid

    def _plot_grid(
        self, name, data, plot_panel, *, prefixed=False, layout_rect=None, **kwargs
    ):
        """Plot a grid figure and save it.

        Parameters
        ----------
        name : str
            Figure name.
        data : dict
            One item per panel, keyed by panel title.
        plot_panel : callable
            Panel plotting function, passed to ``plot_grid``.
        prefixed : bool
            Whether to prefix the figure name.
        layout_rect : tuple or None
            If given, ``tight_layout`` is applied within this rectangle.
        **kwargs
            Keyword arguments of ``plot_grid``, overriding ``grid``.
        """
        with mpl.rc_context(self.rc):
            fig, _ = plot_grid(data, plot_panel, **(self.grid | kwargs))

            if layout_rect is not None:
                fig.tight_layout(rect=layout_rect)

            fig.savefig(self._get_figpath(name, prefixed=prefixed))


class VolumesFiguresRunner(GridFiguresRunner):
    """Figures comparing original, reconstructed and global mesh volumes.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory where figures are saved.
    week_views : dict
        LDDMM-to-global outputs, one per structure.
    subj_colors : dict
        Colors keyed by subject id.
    subj_id : str
        Subject whose volumes are plotted over time.
    state_dir : pathlib.Path or None
        Directory storing the task state. If None, ``figures_dir``.
    rc : dict or None
        Matplotlib rc parameters. If None, ``PAPER_RC``.
    grid : dict or None
        Default keyword arguments of ``plot_grid``. If None, ``PAPER_GRID``.
    labels : dict or None
        Legend labels keyed by mesh id (``original``, ``reconstructed``,
        ``global``), overriding the defaults.
    """

    def __init__(
        self,
        figures_dir,
        week_views,
        subj_colors,
        subj_id,
        state_dir=None,
        rc=None,
        grid=None,
        labels=None,
    ):
        super().__init__(figures_dir, state_dir=state_dir, rc=rc, grid=grid)

        self.week_views = week_views
        self.subj_colors = subj_colors
        self.subj_id = subj_id
        self.labels = {
            "original": "Original mesh",
            "reconstructed": "Reconstructed mesh",
            "global": "Global mesh",
        } | ({} if labels is None else labels)

    @task
    def volume_trends(self):
        """Plot volumes over gestational weeks (``volume_trends``)."""

        def _is_gestational_week(_, time):
            return 0 <= time <= 42

        def _select(view):
            outer_keys = (self.subj_id,)

            dataset = (
                view.dataset.select_outer(outer_keys)
                .filter_keys(_is_gestational_week)
                .map_values(_get_volume)
            )
            local_meshes = (
                view.local_reconstructed_points.select_outer(outer_keys)
                .filter_keys(_is_gestational_week)
                .map_values(_get_volume)
            )
            global_meshes = (
                view.global_points.select_outer(outer_keys)
                .filter_keys(_is_gestational_week)
                .map_values(_get_volume)
            )

            data = {
                "original": dataset,
                "reconstructed": local_meshes,
                "global": global_meshes,
            }
            return data

        self._plot_grid(
            "volume_trends",
            {struct: _select(view) for struct, view in self.week_views.items()},
            plot_layers,
            plot_layer=plot_nested,
            colors=self.subj_colors,
            include_label=False,
            kind="scatter",
            s=10,
            labels=self.labels,
            layer_kwargs={
                "original": {
                    "color": COLORS["original"],
                    "marker": "o",
                    "facecolors": "none",
                    "s": 20,
                    "linewidths": 1.2,
                },
                "reconstructed": {
                    "color": COLORS["reconstructed"],
                    "marker": "x",
                },
                "global": {
                    "color": COLORS["global"],
                    "marker": "s",
                },
            },
            xlabel="Week",
            ylabel="Volume",
            sharex=True,
        )

    @task
    def volume_hist(self):
        """Plot volume histograms across all samples (``volume_hist``)."""

        def _select(view):
            volumes = view.dataset.map_values(_get_volume).flatten().values_list()
            volumes_rec = (
                view.local_reconstructed_points.map_values(_get_volume)
                .flatten()
                .values_list()
            )
            volumes_global = (
                view.global_points.map_values(_get_volume).flatten().values_list()
            )

            data = {
                "original": volumes,
                "reconstructed": volumes_rec,
                "global": volumes_global,
            }
            return data

        self._plot_grid(
            "volume_hist",
            {struct: _select(view) for struct, view in self.week_views.items()},
            plot_hists,
            xlabel="Volume",
            colors=COLORS,
            labels=self.labels,
            **HIST_KWARGS,
        )


class MetricFiguresRunner(GridFiguresRunner):
    """Figures of local and global distances under one metric.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory where figures are saved.
    distances : dict
        Distance evaluations, one per structure.
    metric_name : str
        Display name of the metric, used in labels (e.g. ``"varifold"``).
    prefix : str
        Metric identifier, used as figure name prefix (e.g. ``"var"``).
    state_dir : pathlib.Path or None
        Directory storing the task state. If None, ``figures_dir``.
    rc : dict or None
        Matplotlib rc parameters. If None, ``PAPER_RC``.
    grid : dict or None
        Default keyword arguments of ``plot_grid``. If None, ``PAPER_GRID``.
    labels : dict or None
        Legend labels keyed by series id (``error``, ``local``, ``global``,
        ``within``, ``between``), overriding the defaults.
    """

    def __init__(
        self,
        figures_dir,
        distances,
        metric_name,
        prefix="",
        state_dir=None,
        rc=None,
        grid=None,
        labels=None,
    ):
        super().__init__(
            figures_dir,
            state_dir=state_dir,
            prefix=prefix,
            rc=rc,
            grid=grid,
        )

        self.distances = distances

        within, between = get_group_labels(metric_name)
        self.labels = {
            "error": "LDDMM registration error",
            "local": "Before registration to global template",
            "global": "After registration",
            "within": within,
            "between": between,
        } | ({} if labels is None else labels)
        self.colors = {
            "error": COLORS["reconstructed"],
            "local": COLORS["original"],
            "global": COLORS["global"],
            "within": COLORS["within"],
            "between": COLORS["between"],
        }

    @task
    def dist_rec(self):
        """Plot LDDMM registration error histograms (``rec_hist``)."""

        def _select(item):
            data = {
                "error": item.local_reconstruction_error().apply(np.array),
            }
            return data

        self._plot_grid(
            "rec_hist",
            {struct: _select(dists) for struct, dists in self.distances.items()},
            plot_hists,
            prefixed=True,
            xlabel="Distance",
            colors=self.colors,
            labels=self.labels,
            **HIST_KWARGS,
        )

    @task
    def dist_hist(self):
        """Plot registration error and pairwise distance histograms.

        Saved as ``rec-local-global_hist``.
        """

        def _select(item):
            data = {
                "error": item.local_reconstruction_error().apply(np.array),
                "local": item.local_pairwise().data,
                "global": item.global_pairwise().data,
            }
            return data

        self._plot_grid(
            "rec-local-global_hist",
            {struct: _select(dists) for struct, dists in self.distances.items()},
            plot_hists,
            prefixed=True,
            xlabel="Distance",
            colors=self.colors,
            labels=self.labels,
            **HIST_KWARGS,
        )

    @task
    def dist_comparison(self):
        """Scatter global against local pairwise distances (``local-global``).

        Pairs are colored by whether both samples belong to the same subject.
        """

        def _select(item):
            local = item.local_pairwise()
            global_ = item.global_pairwise()

            return local.select(global_.labels), global_

        def _group_by(key_a, key_b):
            return "within" if is_intra(key_a, key_b) else "between"

        self._plot_grid(
            "local-global",
            {struct: _select(dists) for struct, dists in self.distances.items()},
            lambda pair, **kwargs: plot_dist_comparison(*pair, **kwargs),
            prefixed=True,
            group_by=_group_by,
            colors=self.colors,
            labels=self.labels,
            xlabel="Local distance",
            ylabel="Global distance",
            alpha=0.1,
            sharex=True,
        )

    @task
    def dist_atlases(self):
        """Plot local and global distance-to-atlas histograms (``atlases_hist``)."""

        def _select(item):
            data = {
                "local": item.local_atlas_distance().apply(np.array),
                "global": item.global_atlas_distance().apply(np.array),
            }
            return data

        self._plot_grid(
            "atlases_hist",
            {struct: _select(dists) for struct, dists in self.distances.items()},
            plot_hists,
            prefixed=True,
            xlabel="Distance",
            **HIST_KWARGS,
        )


class MetricVolumeFiguresMerger(BaseFiguresRunner):
    """Merge metric and volume figures into one document.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory containing the figures, where the document is saved.
    state_dir : pathlib.Path or None
        Directory storing the task state. If None, ``figures_dir``.
    prefix : str
        Metric identifier of the merged metric figures (e.g. ``"var"``).
    sep : str
        Separator between prefix and document name.
    subfigures : dict or None
        Keyword arguments of ``add_subfigures``. If None, ``PAPER_SUBFIGURES``.
    document : dict or None
        Keyword arguments of ``make_figure_document``.
        If None, ``PAPER_DOCUMENT``.
    """

    def __init__(
        self,
        figures_dir,
        state_dir=None,
        prefix="",
        sep="_",
        subfigures=None,
        document=None,
    ):
        super().__init__(figures_dir, state_dir=state_dir, prefix=prefix, sep=sep)

        self.subfigures = PAPER_SUBFIGURES if subfigures is None else subfigures
        self.document = PAPER_DOCUMENT if document is None else document

    @task
    def merge_figures(self):
        """Compile metric and volume figures into ``<prefix>-combined``."""
        figure_names = [
            f"{self.prefix}_rec-local-global_hist",
            f"{self.prefix}_local-global",
            "volume_hist",
            "volume_trends",
        ]
        paths = [f"{name}.pdf" for name in figure_names]

        body = add_subfigures(paths, **self.subfigures)
        text = make_figure_document(
            body, figures_path=self.figures_dir, **self.document
        )

        compile_latex(text, self._get_figpath("-combined", prefixed=True))


class MetricsFiguresMerger(BaseFiguresRunner):
    """Merge the distance figures of several metrics into one document.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory containing the figures, where the document is saved.
    prefixes : sequence of str
        Figure name prefixes of the metrics, separator included (e.g. ``"var_"``).
    state_dir : pathlib.Path or None
        Directory storing the task state. If None, ``figures_dir``.
    """

    def __init__(
        self,
        figures_dir,
        prefixes,
        state_dir=None,
    ):
        if state_dir is None:
            state_dir = figures_dir

        super().__init__(state_dir)

        self.figures_dir = figures_dir
        self.prefixes = prefixes

    @task
    def merge_figures(self):
        """Compile the metric figures into ``metrics_combined``."""
        names = ["rec-local-global_hist", "local-global"]
        figure_names = []
        for prefix in self.prefixes:
            figure_names.extend([f"{prefix}{name}" for name in names])
        paths = [f"{name}.pdf" for name in figure_names]

        body = add_subfigures(paths, n_cols=2)
        text = make_figure_document(body, figures_path=self.figures_dir)

        compile_latex(text, self.figures_dir / "metrics_combined.pdf")


class DistmatFiguresRunner(GridFiguresRunner):
    """Figures of local and global pairwise distance matrices.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory where figures are saved.
    distances : dict
        Distance evaluations, one per structure.
    prefix : str or None
        Metric identifier, used as figure name prefix.
    state_dir : pathlib.Path or None
        Directory storing the task state. If None, ``figures_dir``.
    """

    def __init__(self, figures_dir, distances, prefix=None, state_dir=None):
        # not part of the paper figures: keep matplotlib defaults
        super().__init__(
            figures_dir, state_dir=state_dir, prefix=prefix, rc={}, grid={}
        )
        self.distances = distances

    def _plot_distmats(self, name, select):
        """Plot one distance matrix per structure."""
        self._plot_grid(
            name,
            {struct: select(dists) for struct, dists in self.distances.items()},
            plot_distmat,
            prefixed=True,
            show_ticks=False,
            figsize=(8, 18),
            layout_rect=(0.04, 0.03, 0.96, 0.98),
            sharex=True,
        )

    @task
    def global_dist_matrix(self):
        """Plot global pairwise distance matrices (``global-dist-matrix``)."""
        self._plot_distmats("global-dist-matrix", lambda item: item.global_pairwise())

    @task
    def local_dist_matrix(self):
        """Plot local pairwise distance matrices (``local-dist-matrix``)."""
        self._plot_distmats("local-dist-matrix", lambda item: item.local_pairwise())
