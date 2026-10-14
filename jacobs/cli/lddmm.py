from functools import partial

import click

from . import _options


@click.group()
def lddmm():
    """Analyses using LDDMM."""


@lddmm.command()
@_options.outputs_dir
@_options.figures_dir
@click.option("--subdir/--no-subdir", default=True)
@click.option("--n-components", type=click.IntRange(min=2), default=5)
@click.option("--overwrite", is_flag=True)
@_options.plotting
def pca_pairplots(outputs_dir, figures_dir, subdir, n_components, overwrite):
    """Generate PCA pairplots across subcortical structures."""
    from jacobs.core.lddmm.outputs import load_global_deltas
    from jacobs.core.pca import PCAPairplotsRunner
    from jacobs.core.style import get_subject_colors

    if subdir:
        figures_dir = figures_dir / f"pca_{n_components}"

    runner = PCAPairplotsRunner(
        figures_dir,
        partial(load_global_deltas, outputs_dir),
        colors=get_subject_colors(),
        n_components=n_components,
    )
    runner.run(overwrite=overwrite)


@lddmm.command()
@_options.outputs_dir
@click.option("--overwrite", is_flag=True)
def pca_reconstruction(outputs_dir, overwrite):
    """Evaluate PCA reconstruction error across component counts per structure."""
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.lddmm.outputs import (
        find_experiment_dirs,
        load_reconstruction_inputs,
    )
    from jacobs.core.pca import PCAReconstructionEvaluator

    ks = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 60, 80]

    runners = {
        struct: PCAReconstructionEvaluator(
            struct_dir / "pca_reconstruction_check",
            *load_reconstruction_inputs(struct_dir),
            ks=ks,
        )
        for struct, struct_dir in find_experiment_dirs(outputs_dir).items()
    }

    runner = CompositeTaskRunner(
        runners, state_dir=outputs_dir / ".pca_reconstruction_check"
    )
    runner.run(overwrite=overwrite)


@lddmm.command()
@_options.outputs_dir
@click.option(
    "--metric",
    "metrics",
    type=click.Choice(["eucl", "var"]),
    multiple=True,
    default=("eucl", "var"),
    show_default=True,
)
@click.option("--overwrite", is_flag=True)
def post_dists(outputs_dir, metrics, overwrite):
    """Compute Euclidean and/or varifold distances between global representations."""
    from polpo.workflow.lddmm_to_global.distances import (
        EuclideanDistances,
        PersistentEvaluator,
        VarifoldDistances,
    )
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.lddmm.outputs import find_experiment_dirs

    evaluators = {
        "eucl": EuclideanDistances,
        "var": VarifoldDistances,
    }

    experiment_dirs = find_experiment_dirs(outputs_dir)

    runners = {
        metric: CompositeTaskRunner(
            {
                struct: PersistentEvaluator(
                    Evaluator(experiment_dir),
                    experiment_dir / "post_dists" / metric,
                )
                for struct, experiment_dir in experiment_dirs.items()
            },
            state_dir=outputs_dir / ".post_dists" / metric,
        )
        for metric, Evaluator in evaluators.items()
    }

    runner = CompositeTaskRunner(runners, state_dir=outputs_dir / ".post_dists")
    runner.run(tasks=metrics or None, overwrite=overwrite)


@lddmm.command()
@_options.outputs_dir
@_options.figures_dir
@click.option("--subdir/--no-subdir", default=True)
@click.option(
    "--task",
    "tasks",
    multiple=True,
    help=(
        "Figure set to generate: volumes, eucl, var, "
        "lddmm, merge-volume, merge-metrics. Defaults to all."
    ),
)
@click.option("--overwrite", is_flag=True)
@_options.plotting
def validation_figures(outputs_dir, figures_dir, subdir, tasks, overwrite):
    """Generate validation figures for LDDMM-to-global outputs."""
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.lddmm.outputs import load_multi_distances, load_multi_outputs
    from jacobs.core.lddmm.validation import (
        MetricFiguresRunner,
        MetricsFiguresMerger,
        MetricVolumeFiguresMerger,
        VolumesFiguresRunner,
    )
    from jacobs.core.style import get_subject_colors

    if subdir:
        figures_dir = figures_dir / "lddmm"

    metrics = {"eucl": "eucl", "var": "var", "lddmm": "lddmm"}
    structs = ["Thal", "Caud", "Puta", "Pall", "Hipp", "Amyg"]

    outputs = load_multi_outputs(outputs_dir, structs)
    subject_ids = next(iter(outputs.values())).dataset.keys_list()

    runners = {
        "volumes": VolumesFiguresRunner(
            figures_dir,
            outputs,
            get_subject_colors(subject_ids),
            subj_id="01",
            state_dir=figures_dir / ".volumes",
        ),
    }

    for name, metric in metrics.items():
        runner = MetricFiguresRunner(
            figures_dir,
            load_multi_distances(outputs, f"post_dists/{metric}"),
            prefix=metric,
            state_dir=figures_dir / f".{metric}",
        )

        if metric == "lddmm":
            runners[name] = runner.resolve(tasks=("dist_rec", "dist_atlases"))
        else:
            runners[name] = runner.resolve(exclude_tasks=("dist_rec",))

    runners["merge-volume"] = MetricVolumeFiguresMerger(
        figures_dir,
        prefix="var",
        state_dir=figures_dir / ".metrics_volume_combined",
    )
    runners["merge-metrics"] = MetricsFiguresMerger(
        figures_dir,
        ["eucl_", "var_"],
        state_dir=figures_dir / ".metrics_combined",
    )

    CompositeTaskRunner(runners, figures_dir).run(
        tasks=tasks or None,
        overwrite=overwrite,
        continue_on_error=False,
    )


@lddmm.command()
@_options.outputs_dir
@_options.figures_dir
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    help="eucl, var. Defaults to all.",
)
@click.option("--overwrite", is_flag=True)
@_options.plotting
def distmat_figures(outputs_dir, figures_dir, metrics, overwrite):
    """Plot local and global distance matrices per experiment."""
    from polpo.workflow.task import CompositeTaskRunner

    from jacobs.core.lddmm.outputs import load_distances, load_outputs
    from jacobs.core.lddmm.validation import DistmatFiguresRunner

    figures_dir = figures_dir / "lddmm"
    outputs = load_outputs(outputs_dir)

    runners = {
        metric: DistmatFiguresRunner(
            figures_dir,
            load_distances(outputs, f"post_dists/{metric}"),
            prefix=metric,
            state_dir=figures_dir / f".{metric}_distmats",
        )
        for metric in ("eucl", "var")
    }

    CompositeTaskRunner(runners, figures_dir).run(
        tasks=metrics or None, overwrite=overwrite, continue_on_error=False
    )


@lddmm.command()
@_options.outputs_dir
@click.option("--overwrite", is_flag=True)
def bcv_rank_selection(outputs_dir, overwrite):
    """Select PCA rank per structure via bi-cross-validation on global deltas."""
    from functools import partial

    from polpo.bcv.mesh import GroupedMeshRankSelection
    from polpo.workflow.task import CompositeTaskRunner, TaskRunner

    from jacobs.core.bcv import select_rank
    from jacobs.core.lddmm.outputs import find_experiment_dirs

    runners = {
        struct: TaskRunner.from_callable(
            partial(
                select_rank,
                struct,
                struct_dir,
                GroupedMeshRankSelection(center=True),
                results_dir=struct_dir / "rank_selection",
            ),
            state_dir=struct_dir / "rank_selection",
            name="select_rank",
        )
        for struct, struct_dir in find_experiment_dirs(outputs_dir).items()
    }

    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / ".rank_selection",
    )
    runner.run(overwrite=overwrite)


@lddmm.command()
@_options.outputs_dir
@_options.figures_dir
@click.option("--overwrite", is_flag=True)
@_options.plotting
def bcv_rank_selection_figures(outputs_dir, figures_dir, overwrite):
    """Plot rank-selection errors per structure."""
    from functools import partial

    from jacobs.core.bcv import RankSelectionFiguresRunner, load_rank_selection_results

    RankSelectionFiguresRunner(
        figures_dir / "rank_selection",
        partial(load_rank_selection_results, outputs_dir),
    ).run(overwrite=overwrite)
