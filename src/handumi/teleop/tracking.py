"""Tracking-loss debounce and recovery timing shared by real teleop frontends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingRecoveryConfig:
    lost_debounce_s: float = 0.30
    recover_after_s: float = 2.0
    recover_period_s: float = 5.0


class TrackingRecoveryPolicy:
    """State machine for sustained tracking loss and throttled recovery attempts."""

    def __init__(self, config: TrackingRecoveryConfig | None = None) -> None:
        self.config = config or TrackingRecoveryConfig()
        self.tracking_lost_since: float | None = None
        self.tracking_missing_since: float | None = None
        self.last_recovery_attempt: float | None = None

    @property
    def lost(self) -> bool:
        return self.tracking_lost_since is not None

    def reset(self) -> None:
        self.tracking_missing_since = None
        self.tracking_lost_since = None

    def note_missing(self, now: float) -> bool:
        """Return True exactly when missing tracking becomes sustained loss."""
        if self.tracking_missing_since is None:
            self.tracking_missing_since = now
        if self.tracking_lost_since is not None:
            return False
        if now - self.tracking_missing_since < self.config.lost_debounce_s:
            return False
        self.tracking_lost_since = now
        return True

    def lost_for(self, now: float) -> float:
        if self.tracking_lost_since is None:
            return 0.0
        return now - self.tracking_lost_since

    def should_recover(self, now: float) -> bool:
        if self.tracking_lost_since is None:
            return False
        if self.lost_for(now) < self.config.recover_after_s:
            return False
        if (
            self.last_recovery_attempt is not None
            and now - self.last_recovery_attempt < self.config.recover_period_s
        ):
            return False
        self.last_recovery_attempt = now
        return True
