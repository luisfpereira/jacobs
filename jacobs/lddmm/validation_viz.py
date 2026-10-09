import os
from pathlib import Path

import numpy as np
from polpo.dataset import MappedView, NestedKeyMap
from polpo.dataset.plot import plot_nested
from polpo.datasets.jacobs.lddmm_to_global import find_experiment_dirs
from polpo.datasets.jacobs.plot import get_subject_colors, update_mpl_params
from polpo.datasets.jacobs.tabular import get_session_to_week
from polpo.distmat.plot import plot_dist_comparison
from polpo.latex import add_subfigures, compile_latex, make_figure_document
from polpo.neuroimaging.naming import get_subcortical_struct_long_name
from polpo.plot.pyplot import plot_grid, plot_hists, plot_layers
from polpo.workflow.lddmm_to_global.distances import (
    DistanceResults,
    MultiDistanceResults,
)
from polpo.workflow.lddmm_to_global.output import (
    LddmmToGlobalMultiOutput,
    LddmmToGlobalOutput,
)
from polpo.workflow.task import CompositeTaskRunner, TaskRunner, task


def is_intra(key_a, key_b):
    return key_a[1] == key_b[1]


def group_by_intra_inter(key_a, key_b):
    return "within-group" if is_intra(key_a, key_b) else "between-group"


group_colors = {
    "within-group": "blue",
    "between-group": "green",
}

ALPHA = 0.5


def _get_volume(point):
    return point.as_surface().to_polydata().volume


class BaseFiguresRunner(TaskRunner):
    """Base task runner for figure generation."""

    def __init__(self, figures_dir, state_dir=None, prefix=""):
        if state_dir is None:
            state_dir = figures_dir

        super().__init__(state_dir)

        self.figures_dir = figures_dir
        self.prefix = prefix

    def _get_figpath(self, name, prefixed=False):
        prefix = self.prefix if prefixed else ""
        return self.figures_dir / f"{prefix}{name}.pdf"

    def _plot_grid(self, name, data, plot, *, prefixed=False, **kwargs):
        fig, _ = plot_grid(data, plot, **kwargs)
        fig.savefig(self._get_figpath(name, prefixed=prefixed))


class VolumesFiguresRunner(BaseFiguresRunner):
    def __init__(
        self,
        figures_dir,
        week_views,
        subj_colors,
        subj_id,
        state_dir=None,
    ):
        super().__init__(figures_dir, state_dir=state_dir)

        self.week_views = week_views
        self.subj_colors = subj_colors
        self.subj_id = subj_id

    @task
    def volume_trends(self):
        def _is_gestational_week(_, time):
            return 0 <= time <= 42

        def _select(_, view):
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
                "dataset": dataset,
                "local": local_meshes,
                "global": global_meshes,
            }
            return (data,)

        self._plot_grid(
            "volume_trends",
            self.week_views,
            plot_layers,
            plot_fnc=plot_nested,
            select=_select,
            colors=self.subj_colors,
            include_label=False,
            kind="scatter",
            layer_kwargs={
                "dataset": {
                    "marker": "o",
                    "facecolors": "none",
                },
                "local": {
                    "marker": "x",
                },
                "global": {
                    "marker": "s",
                },
            },
            xlabel="Week",
            ylabel="Volume",
        )

    @task
    def volume_hist(self):
        def _select(_, view):
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
                "reconstruction": volumes,
                "local": volumes_rec,
                "global": volumes_global,
            }
            return (data,)

        self._plot_grid(
            "volume_hist",
            self.week_views,
            plot_hists,
            select=_select,
            sharex=False,
            density=True,
            histtype="bar",
            edgecolor="black",
            alpha=ALPHA,
            xlabel="Volume",
        )


class MetricFiguresRunner(BaseFiguresRunner):
    def __init__(
        self,
        figures_dir,
        distances,
        prefix="",
        state_dir=None,
    ):
        super().__init__(
            figures_dir,
            state_dir=state_dir,
            prefix=prefix,
        )

        self.distances = distances

    @task
    def dist_rec(self):
        def _select(_, item):
            data = {
                "reconstruction": item.local_reconstruction_error().apply(np.array),
            }
            return (data,)

        self._plot_grid(
            "rec_hist",
            self.distances,
            plot_hists,
            select=_select,
            prefixed=True,
            sharex=False,
            density=True,
            histtype="bar",
            edgecolor="black",
            xlabel="Distance",
            alpha=ALPHA,
        )

    @task
    def dist_hist(self):
        def _select(_, item):
            data = {
                "reconstruction": item.local_reconstruction_error().apply(np.array),
                "local": item.local_pairwise().data,
                "global": item.global_pairwise().data,
            }
            return (data,)

        self._plot_grid(
            "rec-local-global_hist",
            self.distances,
            plot_hists,
            select=_select,
            prefixed=True,
            sharex=False,
            density=True,
            histtype="bar",
            edgecolor="black",
            xlabel="Distance",
            alpha=ALPHA,
        )

    @task
    def dist_comparison(self):
        def _select(_, item):
            local = item.local_pairwise()
            global_ = item.global_pairwise()

            return local.select(global_.labels), global_

        self._plot_grid(
            "local-global",
            self.distances,
            plot_dist_comparison,
            prefixed=True,
            select=_select,
            group_by=group_by_intra_inter,
            colors=group_colors,
            xlabel="Local distance",
            ylabel="Global distance",
            alpha=0.1,
        )

    @task
    def dist_atlases(self):
        def _select(_, item):
            data = {
                "local": item.local_atlas_distance().apply(np.array),
                "global": item.global_atlas_distance().apply(np.array),
            }
            return (data,)

        self._plot_grid(
            "atlases_hist",
            self.distances,
            plot_hists,
            select=_select,
            prefixed=True,
            sharex=False,
            density=True,
            histtype="bar",
            edgecolor="black",
            xlabel="Distance",
            alpha=ALPHA,
        )


class MetricVolumeFiguresMerger(BaseFiguresRunner):
    @task
    def merge_figures(self):
        figure_names = [
            f"{self.prefix}_rec-local-global_hist",
            f"{self.prefix}_local-global",
            "volume_hist",
            "volume_trends",
        ]
        paths = [f"{name}.pdf" for name in figure_names]

        body = add_subfigures(paths, n_cols=2)
        text = make_figure_document(body, figures_path=self.figures_dir)

        compile_latex(text, self._get_figpath("-combined", prefixed=True))


class MetricsFiguresMerger(BaseFiguresRunner):
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
        names = ["rec-local-global_hist", "local-global"]
        figure_names = []
        for prefix in self.prefixes:
            figure_names.extend([f"{prefix}{name}" for name in names])
        paths = [f"{name}.pdf" for name in figure_names]

        body = add_subfigures(paths, n_cols=2)
        text = make_figure_document(body, figures_path=self.figures_dir)

        compile_latex(text, self.figures_dir / "metrics_combined.pdf")


def load_outputs(outputs_dir, structs):
    experiment_dirs = find_experiment_dirs(outputs_dir)
    key_to_week = NestedKeyMap.from_inner_key_map(get_session_to_week())

    outputs = {}

    for struct in structs:
        struct_long = get_subcortical_struct_long_name(struct)
        dirs = [experiment_dirs[f"{side}_{struct}"] for side in ("L", "R")]

        output = LddmmToGlobalMultiOutput([LddmmToGlobalOutput(path) for path in dirs])

        outputs[struct_long] = output.view.with_key_map(key_to_week)

    return outputs


def load_distances(outputs, distances_folder):
    return {
        struct: MultiDistanceResults(
            {
                side: MappedView(
                    DistanceResults.from_dir(output_.path / distances_folder),
                    output.key_map.flatten(),
                )
                for side, output_ in zip(("L", "R"), output.outputs)
            }
        )
        for struct, output in outputs.items()
    }


if __name__ == "__main__":
    outputs_dir = Path(os.environ["OUTPUTS_DIR"])
    figures_dir = Path(os.environ["FIGURES_DIR"]) / "lddmm"

    metrics = {
        "varifold": ("post_dists/var", "var"),
        "euclidean": ("post_dists/eucl", "eucl"),
        # "lddmm": ("post_dists/lddmm", "lddmm"),
    }

    update_mpl_params()

    structs = ["Thal", "Caud", "Puta", "Pall", "Hipp", "Amyg"]

    outputs = load_outputs(outputs_dir, structs)

    subj_ids = next(iter(outputs.values())).dataset.keys_list()
    subj_colors = get_subject_colors(subj_ids)

    runners = {
        "volumes": VolumesFiguresRunner(
            figures_dir,
            outputs,
            subj_colors,
            subj_id="01",
            state_dir=figures_dir / ".volumes",
        )
    }

    for metric, (distances_folder, short) in metrics.items():
        distances = load_distances(
            outputs,
            distances_folder=distances_folder,
        )

        runner = MetricFiguresRunner(
            figures_dir,
            distances,
            prefix=f"{short}_",
            state_dir=figures_dir / f".{short}",
        )

        if metric == "lddmm":
            runner = runner.resolve(
                tasks=(
                    "dist_rec",
                    "dist_atlases",
                )
            )
        else:
            runner = runner.resolve(
                exclude_tasks=("dist_rec",),
            )

        runners[metric] = runner

    runners["merge-volume"] = MetricVolumeFiguresMerger(
        figures_dir,
        prefix=metrics["varifold"][1],
        state_dir=figures_dir / ".metrics_volume_combined",
    )

    runners["merge-metrics"] = MetricsFiguresMerger(
        figures_dir,
        [f"{short}_" for _, short in metrics.values() if short != "lddmm"],
        state_dir=figures_dir / ".metrics_combined",
    )

    CompositeTaskRunner(runners, figures_dir).run(
        overwrite=True,
        continue_on_error=False,
        tasks=(
            # "volumes",
            "euclidean",
            "varifold",
            # "lddmm",
            # "merge-volume",
            # "merge-metrics",
        ),
    )
