import click

from jacobs.cli import _options


@click.group()
def validation():
    """Validate global representations against local fits."""


@validation.command()
@_options.outputs_dir
@click.option(
    "--metric",
    "metrics",
    type=click.Choice(["eucl", "var"]),
    multiple=True,
    default=("eucl", "var"),
    show_default=True,
)
@_options.overwrite
@_options.raise_on_error
def compute(outputs_dir, metrics, overwrite, raise_on_error):
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
    runner.run(
        tasks=metrics or None,
        overwrite=overwrite,
        raise_on_error=raise_on_error,
    )


@validation.command()
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
def figures(outputs_dir, figures_dir, subdir, tasks, overwrite):
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


@validation.command()
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
def distmats(outputs_dir, figures_dir, metrics, overwrite):
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
