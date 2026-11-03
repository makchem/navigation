"""Core data types shared across the navigation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GpsStatus(Enum):
    OK = "ok"
    JAMMED = "jammed"
    SPOOFED = "spoofed"


class ModuleStatus(Enum):
    FIX = "fix"
    IMU_ONLY = "imu_only"
    UNRELIABLE = "unreliable"


@dataclass
class State:
    """Full 6-DOF state of the aircraft."""
    lat: float          # degrees
    lon: float          # degrees
    alt_m: float        # metres above ground
    roll_rad: float     # radians
    pitch_rad: float    # radians
    yaw_rad: float      # radians (heading, 0 = north, CW positive)
    timestamp_s: float = 0.0


@dataclass
class Fix:
    """Single position estimate from the optical pipeline."""
    lat: float
    lon: float
    heading_rad: float
    sigma_m: float      # horizontal uncertainty in metres
    ok: bool            # False = no fix (matcher failed / low confidence)


@dataclass
class GpsMeasurement:
    """GPS reading — real, simulated, or absent."""
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    speed_mps: float = 0.0
    available: bool = False


@dataclass
class ModuleOutput:
    """What the navigation module sends to the autopilot each step."""
    lat: float
    lon: float
    alt_m: float
    heading_rad: float
    sigma_m: float
    gps_status: GpsStatus = GpsStatus.OK
    module_status: ModuleStatus = ModuleStatus.FIX
    timestamp_s: float = 0.0
