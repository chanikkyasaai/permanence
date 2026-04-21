"""World state data structures and mutation logic."""

from .state import (
    ActionRecord,
    EmployeeState,
    ExternalRelationshipState,
    MutationType,
    ProjectState,
    WorldState,
    WorldStateMutation,
)

from .consequence_engine import ConsequenceEngine

__all__ = [
    "ActionRecord",
    "EmployeeState",
    "ExternalRelationshipState",
    "MutationType",
    "ProjectState",
    "WorldState",
    "WorldStateMutation",
    "ConsequenceEngine",
]
