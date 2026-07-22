# Architecture

`linq_platform/intelligence` generates validated trade candidates.
`linq_platform/backtest` owns simulation and execution semantics.
`linq_platform/risk` owns position sizing and hard limits.
`linq_platform/live` will remain disabled until paper-trading gates pass.
`legacy/phase4_reference` is immutable reference logic used only for regression parity.
