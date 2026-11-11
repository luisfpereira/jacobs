"""Varifold metric shared by the Euclidean, LDDMM and ENIGMA analyses."""

from polpo.surface_mesh.varifold.geometry import VarifoldMetric
from polpo.surface_mesh.varifold.tuning import SigmaFromScale


def instantiate_varifold_metric(
    meshes, object_ratio=0.2, discretization_ratio=2.0, engine="auto", device="cpu"
):
    """Instantiate a varifold metric tuned on the meshes, as in LDDMM-to-global.

    The default ratios match the LDDMM-to-global kernel tuning. Tuned on the
    baseline inputs of all subjects, the width matches the LDDMM runs.

    Parameters
    ----------
    meshes : list of Surface
        Meshes the kernel width is tuned on.
    object_ratio : float
        Multiplicative factor applied to the object scale.
    discretization_ratio : float
        Multiplicative factor applied to the discretization scale.
    engine : str
        Varifold computation engine.
    device : str
        Device used by the engine.

    Returns
    -------
    metric : VarifoldMetric
    """
    sigma = float(
        SigmaFromScale(
            object_ratio=object_ratio, discretization_ratio=discretization_ratio
        )(meshes)
    )
    return VarifoldMetric(sigma=sigma, engine=engine, device=device)
