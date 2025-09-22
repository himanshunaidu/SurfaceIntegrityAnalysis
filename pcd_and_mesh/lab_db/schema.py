import numpy as np
import pandas as pd
import os
import sys

from enum import Enum
from uuid import UUID, uuid4
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass


### Factors
# - Issue Characteristics: Gap Width, Gap Depth, Gap Orientation, Gap Length, Surface Height Difference
# - Device Characteristics: Device Height, Device Angle, Device Speed

# (Future)
# - Additional Issue Characteristics: Gap Slope
# - Environment Characteristics: Lighting Conditions, Weather Conditions (may not be relevant for lab-controlled datasets)

"""
Classes defining the schema for dataset attributes and build plans.
"""
class AttributeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    
    # Required row identifier
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this dataset row (UUID4).",
        json_schema_extra={"example": "6c2a1a1b-2db5-4c0c-8c1a-8aefc7d1d1d1"},
    )
    
    trial_name: str = Field(
        description="Name of the trial or experiment.",
        json_schema_extra={"example": "1_1"},
    )
    
    # Factors (types are enforced; *values* are only suggested)
    gap_width: float = Field(
        ...,
        description="Width of the gap (mm).",
        json_schema_extra={"suggested": {"typical_range_mm": [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0], "unit": "mm"}},
    )
    gap_depth: float = Field(
        ...,
        description="Depth of the gap (mm).",
        json_schema_extra={"suggested": {"typical_range_mm": [10.0, 25.0, 50.0, 100.0], "unit": "mm"}},
    )
    gap_orientation: str = Field(
        ...,
        description="Orientation of the gap relative to motion.",
        json_schema_extra={"suggested": {"values": ["longitudinal", "transverse", "oblique"]}},
    )
    gap_length: float = Field(
        ...,
        description="Length of the gap (mm).",
        json_schema_extra={"suggested": {"typical_range_mm": [0.0, 100.0, 500.0, 1000.0], "unit": "mm"}},
    )
    surface_height_difference: float = Field(
        ...,
        description="Height difference between adjacent surfaces (mm).",
        json_schema_extra={"suggested": {"typical_range_mm": [0.0, 10.0, 15.0, 20.0, 25.0], "unit": "mm"}},
    )
    device_height: float = Field(
        ...,
        description="Height of the top edge of the device above the surface (mm). These will currently be discrete values for simplicity.",
        json_schema_extra={"suggested": {"typical_range_mm": [560, 810.0, 1080.0, 1370.0, 1670.0], "unit": "mm"}},
    )
    device_angle: float = Field(
        ...,
        description="Pitch/tilt angle of the device (degrees). The limits for these will depend on the device height.",
        json_schema_extra={"suggested": {"typical_range_deg": [0.0, 60.0, 90.0], "unit": "deg"}},
    )
    device_speed: float = Field(
        # The device will move from 205 cm away to 45 cm away, with speed decided by the time taken (1 s, 2 s, 4 s).
        ...,
        description="Device speed (m/s).",
        json_schema_extra={"suggested": {"typical_range_mps": [0.4, 0.8, 1.6], "unit": "m/s"}},
    )

@dataclass
class DatasetBuildPlanOverrides:
    gap_width: Optional[Sequence[float]] = None
    gap_depth: Optional[Sequence[float]] = None
    gap_orientation: Optional[Sequence[str]] = None
    gap_length: Optional[Sequence[float]] = None
    surface_height_difference: Optional[Sequence[float]] = None
    device_height: Optional[Sequence[float]] = None
    device_angle: Optional[Sequence[float]] = None
    device_speed: Optional[Sequence[float]] = None

@dataclass
class DatasetBuildPlan:
    """Configuration for producing the CSV."""
    overrides: Optional[DatasetBuildPlanOverrides] = None      # to narrow/step any dimension(s)
    bandit_test_factors: Sequence[str] = ("gap_length", "device_speed")  # which factors to bandit-test (multiple variants each)
    extra_result_columns: Sequence[str] = ("result_score", "result_label", "notes")
    
"""
Constants for building trial grids and dataframes.

Currently, factors is separate from overrides as both can be modified later.
"""
FACTORS = [
    "gap_width",
    "gap_depth",
    "gap_orientation",
    "gap_length",
    "surface_height_difference",
    "device_height",
    "device_angle",
    "device_speed",
]

BASIC_BANDIT_FACTORS = ("gap_length", "device_speed")
BASIC_LONGITUDINAL_OVERRIDES = DatasetBuildPlanOverrides(
    gap_width=[0.0, 2.0, 5.0, 10.0, 20.0, 30.0],
    gap_depth=[10.0, 25.0, 50.0],
    gap_orientation=["longitudinal"],#, "transverse", "oblique"],
    gap_length=[0.0, 100.0, 500.0],
    surface_height_difference=[0.0, 5.0, 10.0, 20.0],
    device_height=[1080.0, 1370.0],
    device_angle=[60.0],
    device_speed=[0.4, 0.8, 1.6]
)