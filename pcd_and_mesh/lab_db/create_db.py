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

from schema import AttributeSchema, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
from schema_utils import pick_suggestion_values, default_angle_plan, get_default_trial_namer
from schema import BASIC_LONGITUDINAL_OVERRIDES, BASIC_BANDIT_FACTORS

def build_trial_grid(
    model: type[AttributeSchema],
    overrides: DatasetBuildPlanOverrides = None,
    bandit_test_factors: Sequence[str] = None
) -> List[Dict[str, Any]]:
    """
    Build a list of factor names and a list of value combinations.
    - `overrides` lets you specify exact sets for certain fields or step a 2-number range.
      e.g., {"device_angle": [0, 15, 30, 45, 60]}
    """
    overrides = overrides or DatasetBuildPlanOverrides()
    bandit_test_factors = bandit_test_factors or []

    factor_values: Dict[str, List[Any]] = {}
    for factor_name in FACTORS:
        # If this is a bandit-test factor, we add random values after the combos are built.
        if factor_name in bandit_test_factors:
            continue        
        vals = list(getattr(overrides, factor_name) or [])
        if not vals:
            field = AttributeSchema.model_fields[factor_name]
            vals = pick_suggestion_values(field)
        
        if not vals:
            raise ValueError(f"Empty suggestion list for field '{factor_name}'.")
        factor_values[factor_name] = vals
    
    combos = []
    for values in itertools.product(*factor_values.values()):
        combos.append(dict(zip(factor_values.keys(), values)))
        
    # Now add random values for bandit-test factors.
    combo_count = len(combos)
    for factor_name in bandit_test_factors:
        vals = list(getattr(overrides, factor_name) or [])
        if not vals:
            field = AttributeSchema.model_fields[factor_name]
            vals = pick_suggestion_values(field)
        
        if not vals:
            raise ValueError(f"Empty suggestion list for bandit-test field '{factor_name}'.")
        
        random_choices = np.random.choice(vals, size=combo_count, replace=True)
        for i, choice in enumerate(random_choices):
            combos[i][factor_name] = choice
    
    return combos

def make_dataframe(
    build_plan: DatasetBuildPlan,
    trial_namer=lambda i, combo: f"{i+1}"
) -> pd.DataFrame:
    """
    Generate a DataFrame of validated trials + blank result columns.
    """
    combos = build_trial_grid(AttributeSchema, build_plan.overrides, build_plan.bandit_test_factors)
    
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

def save_schema(build_plan: DatasetBuildPlan, factors: Sequence[str], bandit_factors: Sequence[str], other_details: dict, out_path: str):
    """
    Save the schema of the DataFrame to a json file.
    """
    schema = AttributeSchema.model_json_schema()
    schema["properties"]["trial_name"]["description"] = "Name of the trial or experiment."
    schema["properties"]["id"]["description"] = "Unique identifier for this dataset row (UUID4)."
    for col in build_plan.extra_result_columns:
        schema["properties"][col] = {
            "title": col,
            "type": ["number", "null"],
            "description": f"Result column '{col}' (float or null)."
        }

    # Add factor details
    schema["factors"] = factors
    schema["bandit_factors"] = bandit_factors

    # Add other details
    schema["other_details"] = other_details    
    
    with open(out_path, "w") as f:
        import json
        json.dump(schema, f, indent=2)
    print(f"Wrote schema to {out_path}")

if __name__ == "__main__":
    plan = DatasetBuildPlan(overrides=BASIC_LONGITUDINAL_OVERRIDES, bandit_test_factors=BASIC_BANDIT_FACTORS)
    
    NAMING_FACTORS = ["gap_width", "gap_depth", "surface_height_difference"]
    OTHER_FACTORS = [f for f in FACTORS if f not in NAMING_FACTORS]
    trial_namer = get_default_trial_namer(plan.overrides, NAMING_FACTORS, OTHER_FACTORS)

    # trial names like "1_1", "1_2", ... based on row index
    df = make_dataframe(plan, trial_namer=trial_namer)
    print(df.head(10))   # quick peek
    out_path = os.path.join(os.path.dirname(__file__), "lab_db.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} trials to {out_path}")
    
    schema_path = os.path.join(os.path.dirname(__file__), "lab_db_schema.json")
    other_details = {
        "naming_function_details": {
            "naming_factors": NAMING_FACTORS,
            "other_factors": OTHER_FACTORS,
            "description": "Trial names are constructed from the indices of the naming factors, with indices of other factors combined into a final index."
        }
    }
    save_schema(plan, NAMING_FACTORS, BASIC_BANDIT_FACTORS, other_details, schema_path)