"""
This script is used to create a lab database for surface integrity analysis.
It uses the schema to define the column structure of the database, and create the necessary table.

Currently, the database is created as a CSV file, but it can be extended to support other formats (e.g., SQLite).
"""
import numpy as np
import pandas as pd
import os
import sys
import itertools
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

from schema import AttributeSchema, DatasetBuildPlan, FACTORS
from utils.schema_utils import pick_suggestion_values, default_angle_plan

def build_trial_grid(
    model: type[AttributeSchema],
    overrides: Optional[Dict[str, Sequence[Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build a list of factor names and a list of value combinations.
    - `overrides` lets you specify exact sets for certain fields or step a 2-number range.
      e.g., {"device_angle": [0, 15, 30, 45, 60]}
    """
    overrides = overrides or {}

    factor_values: Dict[str, List[Any]] = {}
    for factor_name in FACTORS:
        if factor_name in overrides:
            vals = list(overrides[factor_name])
        else:
            field = AttributeSchema.model_fields[factor_name]
            vals = pick_suggestion_values(field)
        
        if not vals:
            raise ValueError(f"Empty suggestion list for field '{factor_name}'.")
        factor_values[factor_name] = vals
    
    combos = []
    for values in itertools.product(*factor_values.values()):
        combos.append(dict(zip(factor_values.keys(), values)))
        
    return combos

def make_dataframe(
    build_plan: DatasetBuildPlan,
    trial_namer=lambda i, combo: f"{i+1}"
) -> pd.DataFrame:
    """
    Generate a DataFrame of validated trials + blank result columns.
    """
    combos = build_trial_grid(AttributeSchema, build_plan.overrides)
    
    rows: List[Dict[str, Any]] = []
    for i, combo in enumerate(combos):
        # Build a candidate row and validate with Pydantic (id auto-fills)
        row = AttributeSchema(trial_name=trial_namer(i, combo), **combo).model_dump()
        # Add empty result columns (kept separate from schema to avoid 'extra' errors)
        for col in build_plan.extra_result_columns:
            row[col] = None
        rows.append(row)
        
    # Put columns in a tidy order: id, trial_name, factors..., results...
    ordered_cols = (
        ["id", "trial_name"]
        + FACTORS
        + list(build_plan.extra_result_columns)
    )
    df = pd.DataFrame(rows)[ordered_cols]
    return df

if __name__ == "__main__":
    plan = DatasetBuildPlan()
    
    # trial names like "1_1", "1_2", ... based on row index
    df = make_dataframe(plan, trial_namer=lambda i, combo: f"1_{i+1}")
    print(df.head(10))   # quick peek
    out_path = os.path.join(os.path.dirname(__file__), "lab_db.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} trials to {out_path}")