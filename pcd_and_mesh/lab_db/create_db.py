"""
This script is used to create a lab database for surface integrity analysis.
It uses the schema to define the column structure of the database, and create the necessary table.

Currently, the database is created as a CSV file, but it can be extended to support other formats (e.g., SQLite).
"""
import numpy as np
import pandas as pd
import os
import sys
import copy
import itertools
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

from schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
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

def create_rows_from_combos(
    combos: List[Dict[str, Any]],
    trial_namer=lambda i, combo: f"{i+1}"
) -> List[Dict[str, Any]]:
    """
    Create validated rows from the list of factor combinations.
    """
    rows: List[Dict[str, Any]] = []
    for i, combo in enumerate(combos):
        # Build a candidate row and validate with Pydantic (id auto-fills)
        row = AttributeSchema(trial_name=trial_namer(i, combo), **combo).model_dump()
        rows.append(row)
    return rows

def make_main_dataframe(
    build_plan: DatasetBuildPlan,
    trial_namer=lambda i, combo: f"{i+1}",
    rows: List[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generate a DataFrame of validated trials + blank result columns.
    """
    if rows is None:
        print("Generating rows again for main DataFrame...")
        combos = build_trial_grid(AttributeSchema, build_plan.overrides, build_plan.bandit_test_factors)
        rows = create_rows_from_combos(combos, trial_namer=trial_namer)

    print("Length of rows before adding extra columns:", len(rows))
    rows_main: List[Dict[str, Any]] = []
    for i, row in tqdm(enumerate(rows), total=len(rows)):
        # Add empty extra columns (kept separate from schema to avoid 'extra' errors)
        row_main = row.copy()
        for col in build_plan.extra_columns:
            row_main[col] = None
        rows_main.append(row_main)

    # Put columns in a tidy order: id, trial_name, factors..., results...
    ordered_cols = (
        [AttributeSchema.Columns.ID.value, AttributeSchema.Columns.TRIAL_NAME.value]
        + FACTORS
        + list(build_plan.extra_columns)
    )
    df = pd.DataFrame(rows_main)[ordered_cols]
    return df

def make_results_dataframe(
    build_plan: DatasetBuildPlan,
    trial_namer=lambda i, combo: f"{i+1}",
    rows: List[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generate a DataFrame of validated trials + blank result columns.
    """
    if rows is None:
        print("Generating rows again for results DataFrame...")
        combos = build_trial_grid(AttributeSchema, build_plan.overrides, build_plan.bandit_test_factors)
        rows = create_rows_from_combos(combos, trial_namer=trial_namer)

    rows_results: List[Dict[str, Any]] = []
    for i, row in tqdm(enumerate(rows), total=len(rows)):
        # Add empty result columns (kept separate from schema to avoid 'extra' errors)
        row_results = row.copy()
        for col in build_plan.result_columns:
            row_results[col] = None
        rows_results.append(row_results)

    # Put columns in a tidy order: id, trial_name, factors..., results...
    ordered_cols = (
        [AttributeSchema.Columns.ID.value, AttributeSchema.Columns.TRIAL_NAME.value]
        + list(build_plan.result_columns)
    )
    df = pd.DataFrame(rows_results)[ordered_cols]
    return df

def save_schema(build_plan: DatasetBuildPlan, factors: Sequence[str], bandit_factors: Sequence[str], other_details: dict, out_path: str):
    """
    Save the schema of the DataFrame to a json file.
    """
    schema = AttributeSchema.model_json_schema()
    schema["properties"]["trial_name"]["description"] = "Name of the trial or experiment."
    schema["properties"]["id"]["description"] = "Unique identifier for this dataset row (UUID4)."
    for col in build_plan.extra_columns:
        schema["properties"][col] = {
            "title": col,
            "type": ["any"],
            "description": f"Extra column '{col}' (any)."
        }
    for col in build_plan.result_columns:
        schema["properties"][col] = {
            "title": col,
            "type": ["any"],
            "description": f"Result column '{col}' (any)."
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

    NAMING_FACTORS = [
        AttributeSchema.Columns.GAP_DEPTH.value, AttributeSchema.Columns.SURFACE_HEIGHT_DIFFERENCE.value, 
        AttributeSchema.Columns.GAP_WIDTH.value, AttributeSchema.Columns.BOARD_PLACEMENT.value
    ]
    # OTHER_FACTORS = [f for f in FACTORS if f not in NAMING_FACTORS]
    OTHER_FACTORS = [AttributeSchema.Columns.DEVICE_SPEED.value] # Only this has two values
    trial_namer = get_default_trial_namer(plan.overrides, NAMING_FACTORS, OTHER_FACTORS)

    # trial names like "1_1", "1_2", ... based on row index
    combos = build_trial_grid(AttributeSchema, plan.overrides, plan.bandit_test_factors)
    print(f"Built {len(combos)} trial combinations.")
    rows = create_rows_from_combos(combos, trial_namer=trial_namer)
    print(f"Created {len(rows)} validated rows.")
    
    # Create main dataframe
    main_df = make_main_dataframe(plan, trial_namer=trial_namer, rows=rows)
    print(main_df.head(5))   # quick peek
    out_path = os.path.join(os.path.dirname(__file__), "lab_db.csv")
    main_df.to_csv(out_path, index=False)
    print(f"Wrote {len(main_df)} trials to {out_path}")
    
    # Create results dataframe
    results_df = make_results_dataframe(plan, trial_namer=trial_namer, rows=rows)
    print(results_df.head(5))   # quick peek
    out_path = os.path.join(os.path.dirname(__file__), "lab_db_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Wrote {len(results_df)} trials to {out_path}")

    schema_path = os.path.join(os.path.dirname(__file__), "lab_db_schema.json")
    other_details = {
        "naming_function_details": {
            "naming_factors": NAMING_FACTORS,
            "other_factors": OTHER_FACTORS,
            "description": "Trial names are constructed from the indices of the naming factors, with indices of other factors combined into a final index."
        }
    }
    save_schema(plan, NAMING_FACTORS, BASIC_BANDIT_FACTORS, other_details, schema_path)