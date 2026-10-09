import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from polpo.datasets.jacobs.lddmm_to_global import (
    find_experiment_dirs,
    get_output_week_view,
)
from polpo.ext.sklearn.adapter import adapt_transform
from polpo.ext.sklearn.compose import TransformedTargetRegressor
from polpo.ext.sklearn.decomposition import PCA
from polpo.ext.sklearn.evaluation import TruncatedCVEvaluator
from polpo.ext.sklearn.regression import MixedLMRegressor
from polpo.surface_mesh.euclidean import EuclideanSurfaces
from polpo.surface_mesh.transform import DeltaTransform
from polpo.workflow.lddmm_to_global.distances import varifold_metric_from_results
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput
from polpo.workflow.task import CompositeTaskRunner, TaskRunner


def prepare_inputs(struct_dir, max_components):
    source = LddmmToGlobalOutput(struct_dir)
    week_views = get_output_week_view(source)

    raw_dataset = week_views.global_points.map_values(
        lambda point: point.as_pv_surface()
    )
    dataset = raw_dataset.filter_keys(lambda _, time: 0.0 < time < 42.0)

    atlas = week_views.global_atlas.template.as_pv_surface()

    estimator = instantiate_estimator(atlas, max_components)
    metrics = instantiate_metrics(source, atlas)

    return dataset, estimator, metrics


def instantiate_estimator(atlas, max_components):
    transform = Pipeline(
        [
            ("delta", adapt_transform(DeltaTransform(atlas))),
            ("pca", PCA(n_components=max_components)),
        ]
    )
    return TransformedTargetRegressor(
        transformer=transform,
        regressor=MultiOutputRegressor(
            MixedLMRegressor(
                'Y ~ time * C(group, Treatment(reference="control"))',
                groups="subject",
                re_formula="1 + time",
            ),
        ),
    )


def collect_convergence(cross_fit_result):
    """Collect MixedLM convergence status for each fold and output."""
    return {
        "converged": [
            [
                bool(regressor.result_.converged)
                for regressor in estimator.regressor_.estimators_
            ]
            for estimator in cross_fit_result.estimators
        ]
    }


def instantiate_metrics(source, atlas):
    return {
        "squared_eucl": EuclideanSurfaces(atlas.faces).metric.squared_dist,
        "squared_varifold": varifold_metric_from_results(
            source.results,
            # backend=backend,
        ).squared_dist,
    }


def is_case(subj_id):
    return int(subj_id[0]) < 2


def prepare_data(dataset):
    groups = (
        dataset.map_items(lambda outer_key, inner_key, value: outer_key)
        .flatten()
        .apply(np.array)
    )

    # TODO: normalize time differently?
    X = (
        dataset.map_items(
            lambda subj_id, time, _: (time / 42.0, "case", subj_id)
            if is_case(subj_id)
            else (time / 42.0, "control", subj_id)
        )
        .flatten()
        .apply(
            pd.DataFrame,
            columns=["time", "group", "subject"],
        )
    )
    y = dataset.flatten().apply(np.array)

    return X, y, groups, dataset.flatten().keys_list()


def evaluate(struct_dir, max_components, results_dir, **evaluation_kwargs):
    dataset, estimator, metrics = prepare_inputs(struct_dir, max_components)

    evaluator = TruncatedCVEvaluator(
        estimator=estimator,
        metrics=metrics,
        **evaluation_kwargs,
    ).fit(dataset)

    evaluator.result_.to_dir(results_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    outputs_dir = Path(os.environ["OUTPUTS_DIR"])

    struct_dirs = find_experiment_dirs(
        outputs_dir,
        long_name=False,
    )

    max_components = 50
    truncations = list(range(1, max_components + 1))

    runners = {
        struct: TaskRunner.from_callable(
            partial(
                evaluate,
                struct_dir,
                max_components=max_components,
                results_dir=struct_dir / "mixed",
                fit_diagnostics=collect_convergence,
                prepare_data=prepare_data,
                truncations=truncations,
            ),
            name="evaluate",
            state_dir=struct_dir / "mixed" / ".rank_selection",
        )
        for struct, struct_dir in struct_dirs.items()
    }

    runner = CompositeTaskRunner(
        runners,
        state_dir=outputs_dir / ".mixed_rank_selection",
    )

    runner.run(
        overwrite=True,
        continue_on_error=True,
    )
