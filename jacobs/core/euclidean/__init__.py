from functools import partial

import numpy as np
from polpo.dataset import NestedDataset
from polpo.ext.numpy.io import load_indexed_arrays, save_indexed_arrays
from polpo.neuroimaging.naming import get_subcortical_struct_long_name
from polpo.surface_mesh import Surface
from polpo.surface_mesh.transform import DeltaTransform

from jacobs.core.paths import find_experiment_dirs
from jacobs.core.enigma import (
    load_registered_dataset,
    load_registered_dataset_with_baselines,
)
from jacobs.core.study import resolve_subcortical_structs


def compute_baseline_deltas(observations, baseline_inputs):
    """Express observations as deltas from each subject's mean baseline."""
    deltas = {}

    for subject, sessions in observations.iter_outer():
        baselines = baseline_inputs.get_outer(subject).values_list()

        baseline = Surface(
            np.mean([mesh.vertices for mesh in baselines], axis=0),
            baselines[0].faces,
        )

        deltas[subject] = sessions.map_values(DeltaTransform(baseline)).data

    return NestedDataset(deltas)


def flatten_vertices(observations):
    """Flatten mesh vertices into feature vectors."""
    return observations.map_values(lambda mesh: mesh.vertices.reshape(-1))


def load_struct_features(
    struct, data_dir=None, subject_ids=None, subtract_baseline=True
):
    """Load PCA features for one structure: baseline deltas or raw vertices."""
    if subtract_baseline:
        return compute_baseline_deltas(
            *load_registered_dataset_with_baselines(struct, data_dir, subject_ids)
        )
    return flatten_vertices(load_registered_dataset(struct, data_dir, subject_ids))


def load_features(
    structs=None, data_dir=None, subject_ids=None, subtract_baseline=True
):
    """Load per-structure PCA features: baseline deltas or raw vertices."""
    return {
        get_subcortical_struct_long_name(struct): load_struct_features(
            struct, data_dir, subject_ids, subtract_baseline
        )
        for struct in resolve_subcortical_structs(structs)
    }


def compute_mean_surface(surfaces):
    """Average the vertices of surfaces that share the same faces."""
    return Surface(np.mean([s.vertices for s in surfaces], axis=0), surfaces[0].faces)


def compute_subject_baselines(baseline_inputs):
    """Average each subject's baseline inputs, keyed by subject."""
    return {
        subject: compute_mean_surface(sessions.values_list())
        for subject, sessions in baseline_inputs.iter_outer()
    }


def compute_atlas(baseline_inputs):
    """Euclidean atlas: the mean of the subject baselines."""
    return compute_mean_surface(
        list(compute_subject_baselines(baseline_inputs).values())
    )


def to_global_meshes(observations, baseline_inputs, atlas):
    """Express observations on the atlas: atlas + (observation - subject baseline)."""
    baselines = compute_subject_baselines(baseline_inputs)

    def to_global(mesh, baseline):
        return Surface(atlas.vertices + mesh.vertices - baseline.vertices, atlas.faces)

    return NestedDataset(
        {
            subject: sessions.map_values(
                partial(to_global, baseline=baselines[subject])
            ).data
            for subject, sessions in observations.iter_outer()
        }
    )


def compute_global_meshes(observations, baseline_inputs):
    """Express observations on a common atlas: atlas + (observation - subject baseline).

    The Euclidean analogue of LDDMM-to-global: a subject's baseline is the mean
    of its baseline inputs, and the atlas is the mean of the baselines.
    """
    atlas = compute_atlas(baseline_inputs)
    return atlas, to_global_meshes(observations, baseline_inputs, atlas)


def load_global_meshes(struct, data_dir=None, subject_ids=None):
    """Load observations as global meshes on a Euclidean atlas."""
    return compute_global_meshes(
        *load_registered_dataset_with_baselines(struct, data_dir, subject_ids)
    )


def _check_features(features):
    valid = ("deltas", "vertices")
    if features not in valid:
        raise ValueError(f"Unknown features {features!r}, expected one of {valid}.")


def load_eval_meshes(struct, features="deltas", data_dir=None, subject_ids=None):
    """Load the meshes evaluated after rank selection.

    Parameters
    ----------
    struct : str
        Subcortical structure.
    features : str
        ``"deltas"`` (global meshes) or ``"vertices"`` (registered meshes).
    data_dir : pathlib.Path or None
        Data directory.
    subject_ids : sequence or None
        Subjects to load. If None, all subjects.

    Returns
    -------
    meshes : NestedDataset
        Evaluated meshes, keyed by subject and session.
    baseline_inputs : NestedDataset
        Registered baseline inputs, keyed by subject and session.
    """
    _check_features(features)

    observations, baseline_inputs = load_registered_dataset_with_baselines(
        struct, data_dir, subject_ids
    )
    if features == "vertices":
        return observations, baseline_inputs

    atlas = compute_atlas(baseline_inputs)
    return to_global_meshes(observations, baseline_inputs, atlas), baseline_inputs


def get_atlas_distance_dir(base_dir):
    return base_dir / "atlas_distance"


def save_atlas_distances(results_dir, atlas, meshes, metric):
    """Compute and save squared distances from meshes to the atlas."""
    save_indexed_arrays(
        results_dir / "squared_varifold.npz",
        keys=meshes.keys_list(),
        data={
            "squared_dist": np.array(
                [metric.squared_dist(mesh, atlas) for mesh in meshes.values_list()]
            )
        },
    )


def load_atlas_rms_distances(outputs_dir):
    """Load the RMS distance to the atlas, keyed by long structure name."""
    rms = {}
    for struct, struct_dir in find_experiment_dirs(outputs_dir, long_name=True).items():
        _, data = load_indexed_arrays(
            get_atlas_distance_dir(struct_dir) / "squared_varifold.npz"
        )
        rms[struct] = np.sqrt(np.mean(data["squared_dist"]))
    return rms
