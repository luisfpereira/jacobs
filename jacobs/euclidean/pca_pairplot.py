import numpy as np
from polpo.dataset import NestedDataset
from polpo.neuroimaging.naming import (
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)
from polpo.surface_mesh import Surface
from polpo.surface_mesh.transform import DeltaTransform

from jacobs.core import get_subject_colors
from jacobs.pca.core import PCAPairplotsRunner

from .core import prepare_inputs


def prepare_deltas(struct, data_dir=None, subject_ids=None):
    """Compute observation deltas relative to subject-specific baselines."""
    observations, baseline_inputs = prepare_inputs(struct, data_dir, subject_ids)

    deltas = {}

    for subject, sessions in observations.iter_outer():
        baselines = baseline_inputs.get_outer(subject).values_list()

        baseline = Surface(
            np.mean([mesh.vertices for mesh in baselines], axis=0),
            baselines[0].faces,
        )

        deltas[subject] = sessions.map_values(DeltaTransform(baseline)).data

    return NestedDataset(deltas)


def make_runner(figures_dir, data_dir=None, subject_ids=None, n_components=5):
    structs = get_all_subcortical_structs(
        only_bilateral=True,
        interleave=True,
    )
    structs = [struct for struct in structs if not struct.endswith("_Accu")]

    deltas = {
        get_subcortical_struct_long_name(struct): prepare_deltas(
            struct, data_dir, subject_ids
        )
        for struct in structs
    }

    return PCAPairplotsRunner(
        deltas=deltas,
        figures_dir=figures_dir,
        n_components=n_components,
        colors=get_subject_colors(),
    )
