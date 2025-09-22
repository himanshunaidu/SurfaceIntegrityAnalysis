from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

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

def derive_ab_pair_from_suggestions(field: Field) -> Tuple[Any, Any]:
    """
    Pick two representative values (A, B) for an AB-tested factor from its suggestions.
    Heuristics:
      - If 'values' list: take first two distinct values.
      - If '*range*' list with >= 2 items: take the first and last entries (endpoints).
    """
    values = pick_suggestion_values(field) or []
    if not values or len(values) == 1:
        raise ValueError("Need at least 2 suggested values to derive A/B variants.")

    # Prefer endpoints to increase contrast
    a, b = values[0], values[-1]
    if a == b and len(values) >= 2:
        a, b = values[0], values[1]
    return a, b