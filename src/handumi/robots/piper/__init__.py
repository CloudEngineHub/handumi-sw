"""Public re-exports for the Piper embodiment.

Typical usage::

    from handumi.robots.registry import load_embodiment

    runtime = load_embodiment("piper")
    solver = runtime.solver_cls()
    sim = runtime.make_sim()
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazily expose IK classes so shared helpers do not require JAX imports."""
    if name == "KinematicsConfig":
        from handumi.robots.kinematics import KinematicsConfig

        return KinematicsConfig
    if name == "KinematicsSolver":
        from .solver import KinematicsSolver

        return KinematicsSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def Sim(**kwargs):
    """Build a Viser sim for Piper (backward-compatible factory)."""
    from handumi.robots.registry import load_embodiment

    return load_embodiment("piper").make_sim(**kwargs)


__all__ = ["KinematicsConfig", "KinematicsSolver", "Sim"]
