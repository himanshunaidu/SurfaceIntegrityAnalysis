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
    
    # The columns enum has to be kept in sync with the fields below.
    class Columns(str, Enum):
        ID = "id"
        TRIAL_NAME = "trial_name"
        GAP_WIDTH = "gap_width"
        GAP_DEPTH = "gap_depth"
        GAP_ORIENTATION = "gap_orientation"
        GAP_LENGTH = "gap_length"
        SURFACE_HEIGHT_DIFFERENCE = "surface_height_difference"
        DEVICE_HEIGHT = "device_height"
        DEVICE_ANGLE = "device_angle"
        DEVICE_SPEED = "device_speed"
        BOARD_PLACEMENT = "board_placement"
    
    # Required row identifier
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this dataset row (UUID4).",
        json_schema_extra={"example": "6c2a1a1b-2db5-4c0c-8c1a-8aefc7d1d1d1"},
        title="id"
    )
    
    trial_name: str = Field(
        description="Name of the trial or experiment.",
        json_schema_extra={"example": "1_1"},
        title="trial_name"
    )
    
    # Factors (types are enforced; *values* are only suggested)
    gap_width: float = Field(
        ...,
        description="Width of the gap (mm).",
        json_schema_extra={"suggested": {"typical_range_mm": [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0], "unit": "mm"}},
        title="gap_width"
    )
    gap_depth: float = Field(
        ...,
        description="Depth of the gap (mm).",
        json_schema_extra={"suggested": {"typical_range_mm": [10.0, 25.0, 50.0, 100.0], "unit": "mm"}},
        title="gap_depth"
    )
    gap_orientation: str = Field(
        ...,
        description="Orientation of the gap relative to motion.",
        json_schema_extra={"suggested": {"values": ["longitudinal", "transverse", "oblique"]}},
        title="gap_orientation"
    )
    gap_length: float = Field(
        ...,
        description="Length of the gap (mm).",
        json_schema_extra={"suggested": {"typical_range_mm": [0.0, 100.0, 500.0, 1000.0], "unit": "mm"}},
        title="gap_length"
    )
    surface_height_difference: float = Field(
        ...,
        description="Height difference between adjacent surfaces (mm).",
        json_schema_extra={"suggested": {"typical_range_mm": [0.0, 10.0, 15.0, 20.0, 25.0], "unit": "mm"}},
        title="surface_height_difference"
    )
    device_height: float = Field(
        ...,
        description="Height of the top edge of the device above the surface (mm). These will currently be discrete values for simplicity.",
        json_schema_extra={"suggested": {"typical_range_mm": [560, 810.0, 1080.0, 1370.0, 1670.0], "unit": "mm"}},
        title="device_height"
    )
    device_angle: float = Field(
        ...,
        description="Pitch/tilt angle of the device (degrees). The limits for these will depend on the device height.",
        json_schema_extra={"suggested": {"typical_range_deg": [0.0, 60.0, 90.0], "unit": "deg"}},
        title="device_angle"
    )
    device_speed: float = Field(
        # The device will move from 205 cm away to 45 cm away, with speed decided by the time taken (1 s, 2 s, 4 s).
        ...,
        description="Device speed (m/s).",
        json_schema_extra={"suggested": {"typical_range_mps": [0.4, 0.8, 1.6], "unit": "m/s"}},
        title="device_speed"
    )
    board_placement: str = Field(
        ...,
        description="Placement of the boards",
        json_schema_extra={"suggested": {"values": ["up", "down"]}},
        title="board_placement"
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
    board_placement: Optional[Sequence[str]] = None

class ResultColumns(str, Enum):
    SINGLE_DEPTH_AVAILABLE = "single_depth_available"
    DEPTH_FUSION_AVAILABLE = "depth_fusion_available"
    POINT_CLOUD_AVAILABLE = "point_cloud_available"
    POLYGON_MESH_AVAILABLE = "polygon_mesh_available"
    
    SINGLE_DEPTH_NUMBER = "single_depth_number"
    DEPTH_FUSION_NUMBER = "depth_fusion_number"
    POINT_CLOUD_NUMBER = "point_cloud_number"
    POLYGON_MESH_NUMBER = "polygon_mesh_number"
    
    SINGLE_DEPTH_RESULT = "single_depth_result"
    DEPTH_FUSION_RESULT = "depth_fusion_result"
    POINT_CLOUD_RESULT = "point_cloud_result"
    POLYGON_MESH_RESULT = "polygon_mesh_result"

@dataclass
class DatasetBuildPlan:
    """Configuration for producing the CSV."""
    overrides: Optional[DatasetBuildPlanOverrides] = None      # to narrow/step any dimension(s)
    bandit_test_factors: Sequence[str] = (
        AttributeSchema.Columns.GAP_LENGTH.value,
        AttributeSchema.Columns.DEVICE_SPEED.value
    )  # which factors to bandit-test (multiple variants each)
    extra_columns: Sequence[str] = (
        "status", "notes"
    )
    result_columns: Sequence[str] = tuple(item.value for item in ResultColumns)

"""
Constants for building trial grids and dataframes.

Currently, factors is separate from overrides as both can be modified later.
"""
FACTORS = [
    AttributeSchema.Columns.GAP_DEPTH.value,
    AttributeSchema.Columns.SURFACE_HEIGHT_DIFFERENCE.value,
    AttributeSchema.Columns.GAP_WIDTH.value,
    AttributeSchema.Columns.GAP_ORIENTATION.value,
    AttributeSchema.Columns.GAP_LENGTH.value,
    AttributeSchema.Columns.DEVICE_HEIGHT.value,
    AttributeSchema.Columns.DEVICE_ANGLE.value,
    AttributeSchema.Columns.DEVICE_SPEED.value,
    AttributeSchema.Columns.BOARD_PLACEMENT.value
]

BASIC_BANDIT_FACTORS = (
    AttributeSchema.Columns.GAP_LENGTH.value, 
    AttributeSchema.Columns.DEVICE_SPEED.value, 
    AttributeSchema.Columns.DEVICE_HEIGHT.value
)
BASIC_LONGITUDINAL_OVERRIDES = DatasetBuildPlanOverrides(
    gap_width=[0.0, 5.0, 10.0, 20.0, 30.0],
    gap_depth=[25.0, 0.0],
    gap_orientation=["longitudinal"],#, "transverse", "oblique"],
    gap_length=[160.0],
    surface_height_difference=[0.0, 5.0, 10.0, 20.0],
    device_height=[810.0],
    device_angle=[60.0],
    device_speed=[0.6] # We will run multiple trials at different speeds, so we ignore this for naming (only 0.6 m/s here)
)

if __name__ == "__main__":
    plan = DatasetBuildPlan(overrides=BASIC_LONGITUDINAL_OVERRIDES, bandit_test_factors=BASIC_BANDIT_FACTORS)
    print(plan)

    # Example of getting the schema
    schema = AttributeSchema.model_json_schema()
    print(schema)
    
    print(AttributeSchema.Columns.GAP_DEPTH.value)