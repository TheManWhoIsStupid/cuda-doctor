# CUDA Doctor Development Rules

1. Read CUDA_DOCTOR_V0.1_SPEC.md before making architectural changes.
2. Keep collectors separate from diagnosis logic.
3. Never add system-modifying behavior in v0.1.
4. Add or update tests for every meaningful behavior change.
5. Do not silently weaken tests to make them pass.
6. Keep Windows and Linux compatibility in mind.
7. Prefer safe failure and diagnostic output over application crashes.
8. Run pytest and lint checks before finishing a task.
