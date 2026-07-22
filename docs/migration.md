# Migration rules

1. Never overwrite the validated reference implementation.
2. Preserve old files under a timestamped backup before moving them.
3. Port one behavior at a time.
4. Run regression and parity tests after every port.
5. Do not enable live execution based solely on backtest results.
