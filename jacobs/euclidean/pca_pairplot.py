import numpy as np
from polpo.dataset import NestedDataset
from polpo.neuroimaging.naming import (
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)
from polpo.surface_mesh import Surface
from polpo.surface_mesh.transform import DeltaTransform

from jacobs.core import get_subject_colors
from jacobs.euclidean.core import (
    load_registered_dataset,
    load_registered_dataset_with_baselines,
)
from jacobs.pca.core import PCAPairplotsRunner


def prepare_deltas(struct, data_dir=None, subject_ids=None):
    """Compute observation deltas relative to subject-specific baselines."""
    observations, baseline_inputs = load_registered_dataset_with_baselines(
        struct, data_dir, subject_ids
    )

    deltas = {}

    for subject, sessions in observations.iter_outer():
        baselines = baseline_inputs.get_outer(subject).values_list()

        baseline = Surface(
            np.mean([mesh.vertices for mesh in baselines], axis=0),
            baselines[0].faces,
        )

        deltas[subject] = sessions.map_values(DeltaTransform(baseline)).data

    return NestedDataset(deltas)


def prepare_vertices(struct, data_dir=None, subject_ids=None):
    """Load registered surfaces as flattened vertex coordinates."""
    observations = load_registered_dataset(struct, data_dir, subject_ids)

    return observations.map_values(lambda mesh: mesh.vertices.reshape(-1))


def make_runner(
    figures_dir,
    data_dir=None,
    subject_ids=None,
    n_components=5,
    subtract_baseline=True,
):
    structs = get_all_subcortical_structs(
        only_bilateral=True,
        interleave=True,
    )
    structs = [struct for struct in structs if not struct.endswith("_Accu")]

    prepare_data = prepare_deltas if subtract_baseline else prepare_vertices

    deltas = {
        get_subcortical_struct_long_name(struct): prepare_data(
            struct, data_dir, subject_ids
        )
        for struct in structs
    }

    return PCAPairplotsRunner(
        data=deltas,
        figures_dir=figures_dir,
        n_components=n_components,
        colors=get_subject_colors(),
    )
