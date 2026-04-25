# Implementation Context

## Status
- Started from spec-only workspace.
- Establishing the PERMANENCE package from the master specification.
- Keeping the architecture strictly layered: data structures, logic, definitions, environment.
- Core environment package is now implemented.
- Focused pytest slice passes: 7/7 tests.

## Decisions
- Use standard-library Python only for the core runtime.
- Keep `step()` and `reset()` info payloads JSON-serializable with a recursive sanitizer.
- Implement explicit task-specific `world_state_init_fn` mappings.
- Keep `ScenarioGenerator` simple with `sample(seed) -> Dict`.
- Use a lightweight training scaffold rather than a GPU-bound loop in this workspace.

## Next Build Steps
1. Expand test coverage if additional spec-level behaviors need to be locked down.
2. Optionally add more task-specific behavioral tests and smoke tests.
3. If the project grows, split the action registry and task bank into smaller files without changing the layer order.
