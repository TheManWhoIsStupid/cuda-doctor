# CUDA Doctor Development Rules

1. Read `.agent-private/specs/CUDA_DOCTOR_V0.1_SPEC.md` before making architectural changes.
2. Keep collectors separate from diagnosis logic.
3. Never add system-modifying behavior in v0.1.
4. Add or update tests for every meaningful behavior change.
5. Do not silently weaken tests to make them pass.
6. Keep Windows and Linux compatibility in mind.
7. Prefer safe failure and diagnostic output over application crashes.
8. Run pytest and lint checks before finishing a task.

## Documentation policy

- Public, long-lived Agent rules belong in `AGENTS.md`, tracked in Git.
- Internal working documents — specifications, fix plans, review notes, and
  unreleased plans — belong under `.agent-private/`, which is git-ignored and
  local only.
- Never commit files under `.agent-private/`, and never use `git add -f` to
  bypass its ignore rule.
