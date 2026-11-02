"""Safety decision for CT28 selective structural repair.

This check is independent of the learning model and patient data.
The predicted relation must be representable by the selected candidate set.
"""


def should_accept(
    proposed_relation: str,
    selected_relation: str,
    compatibility_satisfied: bool,
    consistency: float,
    threshold: float,
) -> bool:
    valid = {"NONE", "PAIR", "JUNCTION", "BOTH"}
    if proposed_relation not in valid or selected_relation not in valid:
        raise ValueError("Unknown structural relation")
    if not 0.0 <= consistency <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("Invalid consistency value or threshold")
    return (
        proposed_relation != "NONE"
        and selected_relation == proposed_relation
        and bool(compatibility_satisfied)
        and consistency >= threshold
    )
