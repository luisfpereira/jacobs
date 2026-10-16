from functools import partial

import numpy as np
from polpo.dataset import NestedDataset
from polpo.datasets.jacobs.mesh import load_dataset as load_surface_dataset
from polpo.ext.numpy.io import load_indexed_arrays, save_indexed_arrays
from polpo.neuroimaging.enigma.mesh import read_geometry
from polpo.neuroimaging.naming import get_subcortical_struct_long_name
from polpo.surface_mesh import Surface
from polpo.surface_mesh.registration import SurfaceRigidRegistration
from polpo.surface_mesh.transform import DeltaTransform
from polpo.surface_mesh.varifold.geometry import VarifoldMetric
from polpo.surface_mesh.varifold.tuning import SigmaFromScale

from jacobs.core.paths import find_experiment_dirs, resolve_data_dir
from jacobs.core.study import (
    filter_pregnancy_weeks,
    get_session_to_week_map,
    resolve_subcortical_structs,
    resolve_subject_ids,
    select_observations_and_atlas_inputs,
)


def read_surface(path):
    return Surface(*read_geometry(path))


def load_path_dataset(struct, data_dir=None, subject_ids=None):
    data_dir = resolve_data_dir(data_dir)
    subject_ids = resolve_subject_ids(subject_ids)

    return load_surface_dataset(
        data_dir=data_dir,
        subject_subset=subject_ids,
        struct_subset=[struct],
        derivative="enigma",
        as_surface=False,
    )[struct]


def register_dataset(dataset):
    target = dataset.flatten().values_list()[0]

    register = SurfaceRigidRegistration(known_correspondences=True).against_same_target(
        target
    )

    return dataset.transform(register)


def load_registered_dataset(struct, data_dir=None, subject_ids=None):
    """Load and rigidly register pregnancy-window observations."""

    data_dir = resolve_data_dir(data_dir)
    paths = load_path_dataset(struct, data_dir, subject_ids)

    session_to_week = get_session_to_week_map(data_dir)
    paths = filter_pregnancy_weeks(paths, session_to_week)

    surfaces = paths.map_values(read_surface)
    return register_dataset(surfaces)


def load_registered_dataset_with_baselines(struct, data_dir=None, subject_ids=None):
    """Prepare rigidly aligned observation and baseline surface datasets.

    Load surfaces, select observations and baseline inputs, and rigidly
    register all meshes to a common reference.

    Parameters
    ----------
    struct : str
        Subcortical structure to load.
    data_dir : path-like
        Root directory containing the dataset.
    subject_ids : array-like
        Subject identifiers to include.

    Returns
    -------
    observations : NestedDataset
        Registered surfaces within the pregnancy window (weeks 0-42),
        retaining all observations from non-pregnancy subjects.
    baseline_inputs : NestedDataset
        Registered surfaces selected for baseline estimation.
    """
    data_dir = resolve_data_dir(data_dir)
    paths = load_path_dataset(struct, data_dir, subject_ids)

    session_to_week = get_session_to_week_map(data_dir)
    observation_paths, baseline_paths = select_observations_and_atlas_inputs(
        paths, session_to_week
    )

    # avoid loading shared observations twice
    selected_paths = observation_paths.merge(baseline_paths, strict=False)

    surfaces = selected_paths.map_values(read_surface)

    registered = register_dataset(surfaces)

    return (
        registered.select_inner(observation_paths.nested_keys()),
        registered.select_inner(baseline_paths.nested_keys()),
    )


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


def compute_global_meshes(observations, baseline_inputs):
    """Express observations on a common atlas: atlas + (observation - subject baseline).

    The Euclidean analogue of LDDMM-to-global: a subject's baseline is the mean
    of its baseline inputs, and the atlas is the mean of the baselines.
    """
    baselines = {
        subject: compute_mean_surface(sessions.values_list())
        for subject, sessions in baseline_inputs.iter_outer()
    }
    atlas = compute_mean_surface(list(baselines.values()))

    def to_global(mesh, baseline):
        return Surface(atlas.vertices + mesh.vertices - baseline.vertices, atlas.faces)

    meshes = NestedDataset(
        {
            subject: sessions.map_values(
                partial(to_global, baseline=baselines[subject])
            ).data
            for subject, sessions in observations.iter_outer()
        }
    )
    return atlas, meshes


def load_global_meshes(struct, data_dir=None, subject_ids=None):
    """Load observations as global meshes on a Euclidean atlas."""
    return compute_global_meshes(
        *load_registered_dataset_with_baselines(struct, data_dir, subject_ids)
    )


def instantiate_varifold_metric(meshes, object_ratio=0.2, engine="auto", device="cpu"):
    """Instantiate a varifold metric tuned on the meshes, as in LDDMM-to-global."""
    sigma = float(SigmaFromScale(object_ratio=object_ratio)(meshes))
    return VarifoldMetric(sigma=sigma, engine=engine, device=device)


def compute_atlas_rms_distance(atlas, meshes, metric):
    """Compute the RMS distance from meshes to the atlas under the metric."""
    return np.sqrt(np.mean([metric.squared_dist(mesh, atlas) for mesh in meshes]))


def compute_atlas_rms_distances(
    structs=None, data_dir=None, subject_ids=None, metric_kwargs=None
):
    """Compute the RMS varifold distance to the Euclidean atlas, keyed by long structure name."""
    metric_kwargs = metric_kwargs or {}

    distances = {}
    for struct in resolve_subcortical_structs(structs):
        atlas, meshes = load_global_meshes(struct, data_dir, subject_ids)
        meshes = meshes.flatten().values_list()
        metric = instantiate_varifold_metric([atlas], **metric_kwargs)

        distances[get_subcortical_struct_long_name(struct)] = (
            compute_atlas_rms_distance(atlas, meshes, metric)
        )
    return distances


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
