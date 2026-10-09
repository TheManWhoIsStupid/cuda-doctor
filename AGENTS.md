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

## GitHub credential security

- Agents may use an already authenticated GitHub CLI (`gh`) only for
  repository operations the user has explicitly approved (for example,
  creating releases or working with pull requests).
- GitHub authentication and authorization must always be completed
  interactively by the user. Agents may initiate the official login flow and
  present the device-code instructions, but must never authorize, approve,
  or substitute for the user.
- Agents must never print, inspect, reveal, export, copy, log, or persist
  GitHub authentication tokens or credentials. Never run `gh auth token`.
- Never read GitHub CLI credential/config files for the purpose of
  extracting credentials, and never read system credential stores, shell
  history, environment variables, or other secret-storage locations to
  discover GitHub tokens.
- Never write GitHub credentials into repository files, `.agent-private/`,
  temporary project files, logs, prompts, or commit messages. Never convert
  an existing authenticated session into a plaintext token file.
- Never create a personal access token unless the user explicitly decides
  to do so outside the Agent workflow.
- Authentication changes, re-authentication, logout, permission expansion,
  and scope changes require explicit user involvement.
- Use the authenticated GitHub session only for the repository operation
  explicitly requested by the user.
- Git operations should continue using the existing SSH configuration.
