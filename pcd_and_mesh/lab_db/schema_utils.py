import os
import glob
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional
import functools
import operator

from schema import AttributeSchema, DatasetBuildPlanOverrides, ResultColumns

def pick_suggestion_values(field: Field) -> Optional[List[Any]]:
    """
    Extract a discrete list of suggested values for a field.
    - If 'values' exists -> use it directly.
    - If '*range*' exists and is a list -> treat it as discrete.
    """
    extra = field.json_schema_extra or {}
    meta = extra.get("suggested", {})
    if not meta:
        return None
    
    if "values" in meta and isinstance(meta["values"], list):
        return list(meta["values"])

    for k, v in meta.items():
        if "range" in k and isinstance(v, list):
            return list(v)

    return None

def default_angle_plan(min_angle: float, max_angle: float, step: float = 30.0) -> List[float]:
    """Fallback plan for a 2-number angle range."""
    count = int((max_angle - min_angle) / step) + 1
    return [min_angle + i * step for i in range(count)]

def get_default_trial_namer(overrides: DatasetBuildPlanOverrides, factors: Sequence[str], other_factors: Sequence[str]) -> lambda i, combo: str:
    """
    Create a default trial namer function based on overrides and bandit_test_factors.
    The namer will include the bandit_test_factors in the name.
    """
    factor_values: Dict[str, List[Any]] = {}
    for factor in factors + list(other_factors):
        vals = list(getattr(overrides, factor) or [])
        if not vals:
            field = AttributeSchema.model_fields[factor]
            vals = pick_suggestion_values(field)
        if not vals:
            raise ValueError(f"Empty suggestion list for field '{factor}'.")
        factor_values[factor] = vals
    
    def namer(i: int, combo: Dict[str, Any]) -> str:
        parts = [f"{factor_values[factor].index(combo[factor])}" for factor in factors if factor not in other_factors]
        name = "-".join(parts)
        other_part = [factor_values[factor].index(combo[factor]) for factor in other_factors if factor in combo]
        other_index = "".join(str(x) for x in other_part) if other_part else None
        if other_index:
            name += f"-{other_index}"
        return name

    return namer

def load_lab_db_frames(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the lab database from CSV files into a pandas DataFrame.
    Currently, there are two filesL: one for main attributes, and one for results.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Database file not found: {dataset_path}")

    db_main_path = os.path.join(dataset_path, "lab_db.csv")
    if not os.path.exists(db_main_path):
        raise FileNotFoundError(f"Database CSV file not found: {db_main_path}")
    df_main = pd.read_csv(db_main_path)
    expected_columns = set(AttributeSchema.Columns.__members__.values())
    missing_columns = expected_columns - set(df_main.columns)
    if missing_columns:
        raise ValueError(f"Database is missing expected columns: {missing_columns}")
    
    db_results_path = os.path.join(dataset_path, "lab_db_results.csv")
    if not os.path.exists(db_results_path):
        raise FileNotFoundError(f"Database results CSV file not found: {db_results_path}")
    df_results = pd.read_csv(db_results_path)
    expected_result_columns = set(ResultColumns.__members__.values())
    missing_result_columns = expected_result_columns - set(df_results.columns)
    if missing_result_columns:
        raise ValueError(f"Database results is missing expected columns: {missing_result_columns}")
    
    return df_main, df_results
