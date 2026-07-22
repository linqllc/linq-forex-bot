# Native Market Memory Migration

Version `v1.0.1` extracts deterministic market-memory primitives from the frozen
`run_market_memory_v1.py` implementation into
`linq_platform.intelligence.market_memory`.

## Native in this release

- configuration
- candle-column normalization and loading
- ATR and body features
- pivot detection
- threshold-hit resolution
- zone-touch helpers
- timestamp/index helpers
- drawdown and trade-summary helpers
- zone status and price formatting

The original script remains frozen as the parity oracle. Each native function is
compared against the reference implementation on deterministic fixtures.

## Still to migrate

- reaction detection
- reaction clustering
- zone construction
- zone trade simulation
- walk-forward orchestration
- CLI/report printing
