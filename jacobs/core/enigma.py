"""Loading of the ENIGMA meshes and analyses that do not depend on a pipeline."""

from functools import cached_property
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from polpo.datasets.jacobs.mesh import load_dataset as load_surface_dataset
from polpo.datasets.jacobs.utils import is_pregnancy_subject
from polpo.ext.numpy.io import load_indexed_arrays, save_indexed_arrays
from polpo.neuroimaging.enigma.mesh import read_geometry
from polpo.surface_mesh import Surface
from polpo.surface_mesh.registration import SurfaceRigidRegistration
from polpo.workflow.task import TaskRunner, task

from jacobs.core.bilateral import plot_bilateral, plot_boxes
from jacobs.core.paths import find_experiment_dirs, resolve_data_dir
from jacobs.core.study import (
    filter_pregnancy_weeks,
    get_session_to_week_map,
    resolve_subject_ids,
    select_observations_and_atlas_inputs,
)
from jacobs.core.style import PAPER_FIGSIZE, PAPER_RC


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


def register_dataset(dataset, target_data=None, random_state=2026):
    """Rigidly align meshes to a target sampled as in LDDMM-to-global.

    Parameters
    ----------
    dataset : NestedDataset
        Meshes to align.
    target_data : NestedDataset or None
        Dataset whose keys the target is sampled from. If None, ``dataset``.
    random_state : int
        Seed of the target sampling. The default matches the LDDMM-to-global
        runs.

    Returns
    -------
    registered : NestedDataset
    """
    target_data = dataset if target_data is None else target_data
    rng = np.random.default_rng(random_state)

    (target_key,) = target_data.flatten().sample(random_state=rng).keys_list()
    target = dataset.flatten().as_dict()[target_key]

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

    selected_paths = observation_paths.merge(baseline_paths, strict=False)

    surfaces = selected_paths.map_values(read_surface)

    # target sampled from the observations, as in LDDMM-to-global
    registered = register_dataset(surfaces, target_data=observation_paths)

    return (
        registered.select_inner(observation_paths.nested_keys()),
        registered.select_inner(baseline_paths.nested_keys()),
    )


def get_noise_floor_dir(base_dir):
    return base_dir / "noise_floor"


def load_noise_floor_inputs(struct, data_dir=None, subject_ids=None):
    """Load the control sessions and the baseline inputs of one structure.

    Parameters
    ----------
    struct : str
        Subcortical structure.
    data_dir : pathlib.Path or None
        Data directory.
    subject_ids : sequence or None
        Subjects to load. If None, all subjects.

    Returns
    -------
    controls : NestedDataset
        Sessions of the control (non-pregnancy) subjects, keyed by subject and
        session.
    baseline_inputs : NestedDataset
        Baseline inputs of all subjects, on which the varifold width is tuned.
    """
    observations, baseline_inputs = load_registered_dataset_with_baselines(
        struct, data_dir, subject_ids
    )
    pregnancy = [
        subject for subject in observations.keys_list() if is_pregnancy_subject(subject)
    ]
    return observations.drop_outer(pregnancy), baseline_inputs


def save_noise_floor_distances(results_dir, controls, metric):
    """Compute and save squared distances between sessions of each control subject.

    Each pair of a control's sessions is rigidly aligned (the second session
    onto the first, with known correspondences) before measuring its distance.

    Parameters
    ----------
    results_dir : pathlib.Path
        Directory where the distances are saved.
    controls : NestedDataset
        Sessions of the control subjects, keyed by subject and session.
    metric : VarifoldMetric
        Metric measuring the distances.
    """
    keys = []
    squared_dists = []
    for subject, sessions in controls.iter_outer():
        for (session_a, mesh_a), (session_b, mesh_b) in combinations(
            sessions.items(), 2
        ):
            aligned_b = SurfaceRigidRegistration(
                known_correspondences=True
            ).fit_transform(mesh_b, mesh_a)

            keys.append((subject, session_a, session_b))
            squared_dists.append(metric.squared_dist(mesh_a, aligned_b))

    results_dir.mkdir(parents=True, exist_ok=True)
    save_indexed_arrays(
        results_dir / "squared_varifold.npz",
        keys=keys,
        data={"squared_dist": np.array(squared_dists)},
    )


def load_subject_distances(struct_dir):
    """Load the RMS distance between sessions of each control subject.

    Returns
    -------
    distances : numpy.ndarray
        One RMS distance per control subject with at least two sessions.
    """
    keys, data = load_indexed_arrays(
        get_noise_floor_dir(struct_dir) / "squared_varifold.npz"
    )

    by_subject = {}
    for (subject, *_), squared_dist in zip(keys, data["squared_dist"]):
        by_subject.setdefault(subject, []).append(squared_dist)

    return np.array([np.sqrt(np.mean(values)) for values in by_subject.values()])


def load_noise_floor(struct_dir):
    """Load the noise floor of one structure.

    The noise floor is the median, across control subjects, of the RMS distance
    between sessions of the same subject.
    """
    return float(np.median(load_subject_distances(struct_dir)))


def load_noise_floors(outputs_dir):
    """Load the noise floor per structure, keyed by long structure name."""
    return {
        struct: load_noise_floor(struct_dir)
        for struct, struct_dir in find_experiment_dirs(
            outputs_dir, long_name=True
        ).items()
    }


def load_all_subject_distances(outputs_dir):
    """Load per-subject RMS distances per structure, keyed by long name."""
    return {
        struct: load_subject_distances(struct_dir)
        for struct, struct_dir in find_experiment_dirs(
            outputs_dir, long_name=True
        ).items()
    }


class NoiseFloorFiguresRunner(TaskRunner):
    """Figures of the distances between sessions of control subjects.

    Parameters
    ----------
    figures_dir : pathlib.Path
        Directory where figures are saved.
    load_distances : callable
        Loads per-subject RMS distances per structure, keyed by long name.
    rc : dict or None
        Matplotlib rc parameters applied while plotting and saving.
        If None, ``PAPER_RC``.
    figsize : tuple[float, float] or None
        Figure size in inches. If None, ``PAPER_FIGSIZE``.
    legend : bool
        Whether to add a legend.
    state_dir : pathlib.Path or None
        Directory storing the task state.
    """

    def __init__(
        self,
        figures_dir,
        load_distances,
        rc=None,
        figsize=None,
        legend=True,
        state_dir=None,
    ):
        super().__init__(state_dir or figures_dir / ".noise_floor")
        self.figures_dir = figures_dir
        self._load_distances = load_distances
        self.rc = PAPER_RC if rc is None else rc
        self.figsize = PAPER_FIGSIZE if figsize is None else figsize
        self.legend = legend

    @cached_property
    def distances(self):
        return self._load_distances()

    @task
    def plot_noise_floor(self):
        """Box plot of per-subject distances, whose medians are the noise floors."""
        with mpl.rc_context(self.rc):
            fig, ax = plt.subplots(figsize=self.figsize, layout="constrained")
            plot_bilateral(ax, self.distances, plot_boxes)

            ax.set_ylabel("Varifold distance")
            if self.legend:
                ax.legend()

            self.figures_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.figures_dir / "noise_floor")
            plt.close(fig)
