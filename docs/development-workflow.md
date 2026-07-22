# Development workflow

1. Update `develop` from GitHub.
2. Create `feature/<short-name>` or `fix/<short-name>`.
3. Make one focused change.
4. Run lint, formatting, tests, and parity checks.
5. Push the branch and open a pull request into `develop`.
6. Merge `develop` into `main` only after the integrated system passes.

## Release rule

A release is not considered validated solely because a backtest is profitable. It must also pass reproducibility, leakage, cost, out-of-sample, walk-forward, and risk-control checks appropriate to the change.
