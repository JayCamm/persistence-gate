from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateProfile:
    """Named operating profile for the Persistence Gate.

    Profiles let downstream systems choose a safety/recall tradeoff without
    rewriting scorer internals.
    """

    name: str
    allow_threshold: float
    warning_threshold: float
    quarantine_threshold: float
    harm_weight: float
    burden_weight: float
    staleness_weight: float
    risk_weight: float
    high_risk_block_threshold: float | None = None
    high_harm_block_threshold: float | None = None


PROFILES: dict[str, GateProfile] = {
    "permissive": GateProfile(
        name="permissive",
        allow_threshold=0.22,
        warning_threshold=0.10,
        quarantine_threshold=-0.18,
        harm_weight=0.45,
        burden_weight=0.25,
        staleness_weight=0.25,
        risk_weight=0.22,
        high_risk_block_threshold=None,
        high_harm_block_threshold=None,
    ),
    "balanced": GateProfile(
        name="balanced",
        allow_threshold=0.28,
        warning_threshold=0.15,
        quarantine_threshold=-0.10,
        harm_weight=0.55,
        burden_weight=0.35,
        staleness_weight=0.35,
        risk_weight=0.28,
        high_risk_block_threshold=None,
        high_harm_block_threshold=0.85,
    ),
    "conservative": GateProfile(
        name="conservative",
        allow_threshold=0.34,
        warning_threshold=0.20,
        quarantine_threshold=-0.05,
        harm_weight=0.72,
        burden_weight=0.40,
        staleness_weight=0.50,
        risk_weight=0.42,
        high_risk_block_threshold=0.80,
        high_harm_block_threshold=0.70,
    ),
}


def get_profile(profile: str | GateProfile = "balanced") -> GateProfile:
    if isinstance(profile, GateProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown gate profile {profile!r}. Valid profiles: {valid}") from exc
