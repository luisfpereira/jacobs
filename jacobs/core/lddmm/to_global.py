import logging

from polpo.dataset import NestedKeyEncoder
from polpo.workflow.lddmm_to_global import LddmmToGlobal

from jacobs.core.paths import resolve_data_dir
from jacobs.core.study import (
    get_session_to_week_map,
    resolve_subject_ids,
    select_observations_and_atlas_inputs,
)

logger = logging.getLogger(__name__)


def load_to_global_inputs(struct, subject_ids, data_dir, derivative="enigma"):
    """Prepare mesh data for an LDDMM-to-global run.

    Load meshes for one subcortical structure, select observations used for
    subject-specific atlas estimation, discard subjects without suitable atlas
    inputs, select observations in the target gestational window, and encode
    dataset keys for use in output paths.

    Parameters
    ----------
    struct : str
        Subcortical structure to process.
    subject_ids : array-like
        Subject identifiers to include.
    data_dir : path-like
        Root directory containing the dataset.
    derivative : str
        Mesh derivative to load.

    Returns
    -------
    dataset : NestedDataset
        Mesh dataset with encoded subject-session keys.
    atlas_keys : mapping
        Encoded subject-session keys identifying observations used for atlas
        estimation.
    atlas_only_keys : mapping
        Encoded subject-session keys used for atlas estimation but excluded
        from the global representations.
    known_correspondences : bool
        Whether the loaded meshes have known vertex correspondences.
    metadata : dict
        Metadata describing the prepared inputs, including the original-to-
        encoded key mapping.
    """
    from polpo.datasets.jacobs.mesh import load_dataset

    dataset = load_dataset(
        data_dir=data_dir,
        subject_subset=subject_ids,
        struct_subset=[struct],
        derivative=derivative,
        as_surface=True,
    )[struct]

    session_to_week = get_session_to_week_map(data_dir)
    observations, atlas_inputs = select_observations_and_atlas_inputs(
        dataset, session_to_week
    )

    inputs = observations.merge(atlas_inputs, strict=False)
    atlas_only_inputs = atlas_inputs.drop_inner(observations.nested_keys())

    # encode dataset keys for manageable folder names
    key_codec = NestedKeyEncoder()(inputs.nested_keys())

    metadata = dict(
        struct=struct,
        subject_ids=subject_ids,
        derivative=derivative,
        data_dir=data_dir.as_posix(),
        key_map=key_codec.to_dict(),
    )

    return (
        inputs.map_keys(key_codec.map),
        key_codec.map_keys(atlas_inputs.nested_keys()),
        key_codec.map_keys(atlas_only_inputs.nested_keys()),
        derivative == "enigma",
        metadata,
    )


def run_to_global(
    struct,
    results_dir,
    subject_ids=None,
    data_dir=None,
    derivative="enigma",
    random_state=2026,
    **protocol_kwargs,
):
    """Run LDDMM-to-global for one subcortical structure."""
    dataset, atlas_keys, atlas_only_keys, known_correspondences, metadata = (
        load_to_global_inputs(
            struct,
            resolve_subject_ids(subject_ids),
            resolve_data_dir(data_dir),
            derivative,
        )
    )

    protocol = LddmmToGlobal(
        results_dir=results_dir,
        metadata=metadata,
        known_correspondences=known_correspondences,
        random_state=random_state,
        **protocol_kwargs,
    )
    protocol.run(dataset, atlas_keys=atlas_keys, atlas_only_keys=atlas_only_keys)
