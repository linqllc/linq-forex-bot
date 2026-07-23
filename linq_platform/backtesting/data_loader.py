"""CSV loaders for historical candles and trade signals."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from linq_platform.backtesting.models import (
    Candle,
    OrderType,
    TradeDirection,
    TradeSignal,
)


_CANDLE_ALIASES = {
    "time": ("time", "timestamp", "datetime", "date"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "volume": ("volume", "vol", "tick_volume"),
}

_SIGNAL_ALIASES = {
    "signal_id": ("signal_id", "id"),
    "time": ("time", "timestamp", "datetime", "date"),
    "direction": ("direction", "side"),
    "entry_price": ("entry_price", "entry"),
    "stop_price": ("stop_price", "stop", "stop_loss"),
    "target_price": (
        "target_price",
        "target",
        "take_profit",
    ),
    "order_type": ("order_type", "type"),
}


def _normalized_headers(
    fieldnames: list[str] | None,
) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV file must include a header row.")

    return {field.strip().lower(): field for field in fieldnames}


def _find_column(
    headers: dict[str, str],
    aliases: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    for alias in aliases:
        if alias in headers:
            return headers[alias]

    if required:
        raise ValueError("Missing required CSV column. Expected one of: " + ", ".join(aliases))

    return None


def _parse_datetime(value: str) -> datetime:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("Timestamp cannot be empty.")

    normalized = cleaned.replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y.%m.%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    )

    for format_string in formats:
        try:
            return datetime.strptime(
                cleaned,
                format_string,
            )
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp format: {value!r}")


def _parse_float(
    value: str,
    *,
    field_name: str,
) -> float:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")

    try:
        return float(cleaned)
    except ValueError as error:
        raise ValueError(f"Invalid {field_name}: {value!r}") from error


def load_candles_csv(
    path: str | Path,
) -> list[Candle]:
    """Load validated OHLCV candles from a CSV file."""

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(f"Candle CSV not found: {source}")

    candles: list[Candle] = []

    with source.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        headers = _normalized_headers(reader.fieldnames)

        columns = {
            key: _find_column(
                headers,
                aliases,
                required=key != "volume",
            )
            for key, aliases in _CANDLE_ALIASES.items()
        }

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                volume_column = columns["volume"]
                volume = (
                    _parse_float(
                        row[volume_column],
                        field_name="volume",
                    )
                    if (volume_column is not None and row.get(volume_column, "").strip())
                    else 0.0
                )

                candles.append(
                    Candle(
                        time=_parse_datetime(row[columns["time"]]),
                        open=_parse_float(
                            row[columns["open"]],
                            field_name="open",
                        ),
                        high=_parse_float(
                            row[columns["high"]],
                            field_name="high",
                        ),
                        low=_parse_float(
                            row[columns["low"]],
                            field_name="low",
                        ),
                        close=_parse_float(
                            row[columns["close"]],
                            field_name="close",
                        ),
                        volume=volume,
                    )
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(f"Invalid candle row {row_number}: {error}") from error

    if not candles:
        raise ValueError("Candle CSV contains no data rows.")

    previous_time: datetime | None = None

    for candle in candles:
        if previous_time is not None and candle.time <= previous_time:
            raise ValueError("Candle timestamps must be strictly increasing with no duplicates.")

        previous_time = candle.time

    return candles


def _parse_direction(
    value: str,
) -> TradeDirection:
    normalized = value.strip().lower()

    aliases = {
        "long": TradeDirection.LONG,
        "buy": TradeDirection.LONG,
        "short": TradeDirection.SHORT,
        "sell": TradeDirection.SHORT,
    }

    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"Invalid trade direction: {value!r}") from error


def _parse_order_type(
    value: str | None,
) -> OrderType:
    if value is None or not value.strip():
        return OrderType.MARKET

    normalized = value.strip().lower()

    try:
        return OrderType(normalized)
    except ValueError as error:
        raise ValueError(f"Invalid order type: {value!r}") from error


def load_signals_csv(
    path: str | Path,
) -> list[TradeSignal]:
    """Load validated trade signals from a CSV file."""

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(f"Signal CSV not found: {source}")

    signals: list[TradeSignal] = []

    with source.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        headers = _normalized_headers(reader.fieldnames)

        columns = {
            key: _find_column(
                headers,
                aliases,
                required=key
                not in {
                    "target_price",
                    "order_type",
                },
            )
            for key, aliases in _SIGNAL_ALIASES.items()
        }

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                target_column = columns["target_price"]
                order_type_column = columns["order_type"]

                target_value = row.get(target_column, "") if target_column is not None else ""

                metadata: dict[str, Any] = {"source_row": row_number}

                signals.append(
                    TradeSignal(
                        signal_id=row[columns["signal_id"]].strip(),
                        time=_parse_datetime(row[columns["time"]]),
                        direction=_parse_direction(row[columns["direction"]]),
                        entry_price=_parse_float(
                            row[columns["entry_price"]],
                            field_name="entry_price",
                        ),
                        stop_price=_parse_float(
                            row[columns["stop_price"]],
                            field_name="stop_price",
                        ),
                        target_price=(
                            _parse_float(
                                target_value,
                                field_name=("target_price"),
                            )
                            if target_value.strip()
                            else None
                        ),
                        order_type=_parse_order_type(
                            row.get(order_type_column) if order_type_column is not None else None
                        ),
                        metadata=metadata,
                    )
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(f"Invalid signal row {row_number}: {error}") from error

    return signals
