# Contributing

## Branches

- `main`: stable, releasable, and passing CI.
- `develop`: integration branch for completed features.
- `feature/<name>`: new work.
- `fix/<name>`: bug fixes.

Do not develop directly on `main`.

## Local checks

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

To apply formatting:

```bash
python -m ruff format .
```

## Pull requests

Open feature and fix pull requests into `develop`. Promote tested releases from `develop` into `main`.

Any change affecting signal generation, fills, costs, risk, or execution must include regression coverage and clearly state whether historical parity is expected to change.
