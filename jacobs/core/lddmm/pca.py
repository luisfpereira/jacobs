import numpy as np
from polpo.ext.sklearn.decomposition import PCA
from polpo.workflow.lddmm_to_global.distances import DistanceResults
from polpo.workflow.lddmm_to_global.output import LddmmToGlobalOutput

from jacobs.core.bcv import load_selected_rank
from jacobs.core.pca import get_pca_reconstruction_eval_dir, load_reconstruction_errors

from .outputs import find_experiment_dirs, load_global_deltas


def load_reconstruction_summary(output):
    """Collect sweep errors, error scales, BCV rank and PCA variance for one experiment."""
    struct_dir = output.path
    results_dir = get_pca_reconstruction_eval_dir(struct_dir)

    sdists = {}
    for name in ("varifold", "euclidean"):
        ks, sdists[name] = load_reconstruction_errors(results_dir, f"squared_{name}")

    error_scale = {}
    template_mse = {}
    for name, metric in {"varifold": "var", "euclidean": "eucl"}.items():
        results = DistanceResults.from_dir(struct_dir / "post_dists" / metric)
        error_scale[name] = (
            results.local_reconstruction_error().apply(lambda x: np.stack(x).mean())
            ** 2
        )
        template_mse[name] = results.global_atlas_distance().apply(
            lambda x: (np.stack(x) ** 2).mean()
        )

    deltas = load_global_deltas(output).flatten().values_list()
    pca = PCA(n_components=max(ks)).fit(np.stack(deltas))

    return {
        "ranks": ks,
        "sdists": sdists,
        "error_scale": error_scale,
        "template_mse": template_mse,
        "bcv_rank": load_selected_rank(struct_dir),
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def load_reconstruction_summaries(outputs_dir):
    return {
        struct: load_reconstruction_summary(LddmmToGlobalOutput(struct_dir))
        for struct, struct_dir in find_experiment_dirs(
            outputs_dir, long_name=True
        ).items()
    }
