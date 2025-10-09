"""
This script is used to calculate attributes and metrics for a lab database or a given row in the lab database.
"""
import os
import pandas as pd
import os
import sys
import copy
import itertools
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

if __name__ == "__main__":
    from schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
    from schema_utils import pick_suggestion_values, default_angle_plan, get_default_trial_namer
    from schema import BASIC_LONGITUDINAL_OVERRIDES, BASIC_BANDIT_FACTORS
else:
    if __package__ is None or __package__ == "":
        # Assuming running as a script from the parent directory
        from db.schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
        from db.schema_utils import pick_suggestion_values, default_angle_plan, get_default_trial_namer
        from db.schema import BASIC_LONGITUDINAL_OVERRIDES, BASIC_BANDIT_FACTORS
    else:
        from .schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
        from .schema_utils import pick_suggestion_values, default_angle_plan, get_default_trial_namer
        from .schema import BASIC_LONGITUDINAL_OVERRIDES, BASIC_BANDIT_FACTORS

def calc_integrity_issue(main_frame: pd.DataFrame, index: int) -> bool:
    """
    Calculate the presence of integrity issues based on the row attributes.
    """
    row: pd.Series = main_frame.iloc[index]
    
    gap_width = row[AttributeSchema.Columns.GAP_WIDTH.value]
    surface_height_difference = row[AttributeSchema.Columns.SURFACE_HEIGHT_DIFFERENCE.value]
    
    return gap_width > 0 or surface_height_difference > 0