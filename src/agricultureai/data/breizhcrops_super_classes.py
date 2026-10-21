"""BreizhCrops raw class ids (0..8) mapped to Level-1 DMOE super-classes (0..3)."""

from __future__ import annotations

from typing import Dict, Iterable, List

RAW_TO_SUPER_CLASS: Dict[int, int] = {
    0: 0,
    1: 0,  # barley, wheat -> Winter Cereals
    2: 1,  # rapeseed
    3: 2,  # corn
    4: 3,
    5: 3,
    6: 3,  # sunflower, orchards, nuts -> Grasslands (Level-1)
    7: 3,
    8: 3,  # permanent/temporary meadows -> Grasslands
}

SUPER_CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "Winter Cereals",
    1: "Rapeseed",
    2: "Corn",
    3: "Grasslands",
}


def map_raw_to_super_class(label: int) -> int:
    """Map a raw BreizhCrops class id (0..8) to a super-class id (0..3)."""
    try:
        return RAW_TO_SUPER_CLASS[int(label)]
    except KeyError as exc:
        raise ValueError(f"Unknown raw BreizhCrops class id: {label}") from exc


def raw_class_ids_for_super_targets(super_ids: Iterable[int]) -> List[int]:
    """Expand super-class target ids to the raw BreizhCrops ids they contain."""
    super_set = {int(x) for x in super_ids}
    return sorted(
        raw_id for raw_id, super_id in RAW_TO_SUPER_CLASS.items() if super_id in super_set
    )
