"""Canonical human-body representation and source mappings."""

from handumi.body.mapping import canonical_body_from_packet
from handumi.body.model import (
    CANONICAL_BODY_SCHEMA,
    CANONICAL_JOINT_COUNT,
    CANONICAL_JOINTS,
    PICO_BODY_24_SOURCE_NAMES,
    CanonicalBodyFrame,
    CanonicalJoint,
    CanonicalProvenance,
    canonical_body_features,
    canonical_body_metadata,
)

__all__ = [
    "CANONICAL_BODY_SCHEMA",
    "CANONICAL_JOINT_COUNT",
    "CANONICAL_JOINTS",
    "PICO_BODY_24_SOURCE_NAMES",
    "CanonicalBodyFrame",
    "CanonicalJoint",
    "CanonicalProvenance",
    "canonical_body_features",
    "canonical_body_from_packet",
    "canonical_body_metadata",
]
