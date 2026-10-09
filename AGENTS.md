# CUDA Doctor Development Rules

1. Understand the existing architecture before making architectural changes;
   this file, the README, and the code are the authoritative references.
2. Keep collectors separate from diagnosis logic.
3. Never add system-modifying behavior in v0.1.
4. Add or update tests for every meaningful behavior change.
5. Do not silently weaken tests to make them pass.
6. Keep Windows and Linux compatibility in mind.
7. Prefer safe failure and diagnostic output over application crashes.
8. Run pytest and lint checks before finishing a task.

## Documentation policy

- Public, long-lived Agent rules belong in `AGENTS.md`, tracked in Git. This
  file is fully self-contained: following it never requires reading anything
  under `.agent-private/`.
- Internal working documents — specifications, fix plans, review notes, and
  unreleased plans — belong under `.agent-private/`, which is git-ignored and
  local only.
- `.agent-private/` content is optional, maintainer-only context: maintainers
  may consult those files when they exist, but nothing in this repository may
  require them. External contributors and normal repository use rely only on
  `AGENTS.md`, the README, and the code.
- Never commit files under `.agent-private/`, and never use `git add -f` to
  bypass its ignore rule.
