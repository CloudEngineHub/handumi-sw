"""Repaint the context camera so it shows the target robot, not the operator.

HandUMI's wrist cameras are already embodiment-agnostic; the context camera is
not, because it sees the operator's arms. This package rewrites that stream for
a chosen embodiment, one clip at a time, with the verification and budget
discipline the work needs: a generative edit is not reproducible the way a
renderer is, so every output records what produced it.
"""

from handumi.inpainting.clip import (
    ClipSpec,
    align_to_source,
    extract_clip,
    extract_reference,
    resolve_episode_clip,
)
from handumi.inpainting.compositing import (
    MaskConfig,
    composite,
    edit_mask,
    operator_footprint,
)
from handumi.inpainting.dataset import (
    WORKSPACE_VIDEO_KEY,
    InpaintedDatasetReport,
    write_inpainted_dataset,
)
from handumi.inpainting.extrinsics import (
    CameraFromTable,
    MarkerConfig,
    detect_marker,
    retarget_offset_px,
    solve_camera_from_table,
)
from handumi.inpainting.gates import (
    GateFinding,
    GateReport,
    GateThresholds,
    evaluate,
    read_video,
)
from handumi.inpainting.ledger import (
    DEFAULT_MAX_CALLS,
    Budget,
    Ledger,
    file_sha256,
    now_iso,
)
from handumi.inpainting.omni import EditResult, edit_clip
from handumi.inpainting.prompt import load_prompt, prompt_path

__all__ = [
    "DEFAULT_MAX_CALLS",
    "Budget",
    "CameraFromTable",
    "ClipSpec",
    "EditResult",
    "GateFinding",
    "GateReport",
    "GateThresholds",
    "Ledger",
    "WORKSPACE_VIDEO_KEY",
    "InpaintedDatasetReport",
    "MarkerConfig",
    "MaskConfig",
    "align_to_source",
    "composite",
    "detect_marker",
    "edit_clip",
    "edit_mask",
    "evaluate",
    "extract_clip",
    "extract_reference",
    "file_sha256",
    "load_prompt",
    "now_iso",
    "operator_footprint",
    "prompt_path",
    "read_video",
    "write_inpainted_dataset",
    "resolve_episode_clip",
    "retarget_offset_px",
    "solve_camera_from_table",
]
