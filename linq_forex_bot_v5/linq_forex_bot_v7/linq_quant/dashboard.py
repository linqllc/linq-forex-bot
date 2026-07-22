from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from linq_quant.confidence import sample_warning


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []

    result = frame.copy()

    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].round(4)

    return result.where(
        pd.notna(result),
        None,
    ).to_dict(orient="records")


def build_dashboard(
    output_path: Path,
    setups: pd.DataFrame,
    pair_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
    weekday_summary: pd.DataFrame,
    ab_summary: pd.DataFrame,
    monte_carlo_summary: dict[str, object] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = setups.dropna(subset=["final_r"]).copy()

    metadata = {
        "pairs": sorted(
            completed["instrument"].dropna().astype(str).unique()
        ),
        "sessions": sorted(
            completed["session"].dropna().astype(str).unique()
        ),
        "sources": sorted(
            completed["data_source"].dropna().astype(str).unique()
        ),
        "versions": sorted(
            completed["strategy_version"].dropna().astype(str).unique()
        ),
        "timeframes": sorted(
            completed["timeframe"].dropna().astype(str).unique()
        ),
    }

    dataset_json = json.dumps(
        _records(completed),
        separators=(",", ":"),
    )

    monte_carlo_json = json.dumps(
        monte_carlo_summary or {},
        separators=(",", ":"),
    )

    initial_warning = sample_warning(len(completed))

    document = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LINQ Quant Research Dashboard</title>
<style>
:root {{
    color-scheme: dark;
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 28px;
    background: #111827;
    color: #f9fafb;
}}
h1, h2 {{
    margin-top: 0;
}}
section {{
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
}}
.filters {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 12px;
}}
label {{
    color: #d1d5db;
    font-size: 13px;
}}
select {{
    display: block;
    width: 100%;
    margin-top: 6px;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #4b5563;
    background: #111827;
    color: #f9fafb;
}}
.metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 12px;
}}
.metric {{
    background: #111827;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 14px;
}}
.metric span {{
    display: block;
    color: #9ca3af;
    font-size: 13px;
}}
.metric strong {{
    display: block;
    margin-top: 6px;
    font-size: 22px;
}}
.warning {{
    border-left: 4px solid #f59e0b;
    background: #292313;
    color: #fde68a;
    padding: 14px;
    border-radius: 8px;
    margin-top: 14px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
th, td {{
    border-bottom: 1px solid #374151;
    padding: 9px;
    text-align: right;
}}
th:first-child, td:first-child {{
    text-align: left;
}}
.empty {{
    color: #9ca3af;
}}
</style>
</head>
<body>
<h1>LINQ Quant Research Dashboard</h1>

<section>
<h2>Research Filters</h2>
<div class="filters">
    <label>
        Data source
        <select id="sourceFilter"></select>
    </label>
    <label>
        Pair
        <select id="pairFilter"></select>
    </label>
    <label>
        Session
        <select id="sessionFilter"></select>
    </label>
    <label>
        Strategy version
        <select id="versionFilter"></select>
    </label>
    <label>
        Timeframe
        <select id="timeframeFilter"></select>
    </label>
</div>
</section>

<section>
<h2>Overview</h2>
<div class="metrics">
    <div class="metric">
        <span>Completed trades</span>
        <strong id="tradeCount">0</strong>
    </div>
    <div class="metric">
        <span>Net R</span>
        <strong id="netR">0</strong>
    </div>
    <div class="metric">
        <span>Expectancy R</span>
        <strong id="expectancyR">0</strong>
    </div>
    <div class="metric">
        <span>Win rate</span>
        <strong id="winRate">0%</strong>
    </div>
    <div class="metric">
        <span>Profit factor</span>
        <strong id="profitFactor">0</strong>
    </div>
</div>

<div id="sampleWarning" class="warning">
{html.escape(initial_warning)}
</div>
</section>

<section>
<h2>Pair Performance</h2>
<div id="pairTable"></div>
</section>

<section>
<h2>Session Performance</h2>
<div id="sessionTable"></div>
</section>

<section>
<h2>Weekday Performance</h2>
<div id="weekdayTable"></div>
</section>

<section>
<h2>Multi-Timeframe A/B Comparison</h2>
<div id="abTable"></div>
</section>

<section>
<h2>Monte Carlo Risk</h2>
<pre id="monteCarlo"></pre>
</section>

<script>
const trades = {dataset_json};
const metadata = {json.dumps(metadata)};
const monteCarlo = {monte_carlo_json};

function fillSelect(elementId, values, preferred = "all") {{
    const element = document.getElementById(elementId);
    element.innerHTML = "";

    const all = document.createElement("option");
    all.value = "all";
    all.textContent = "All";
    element.appendChild(all);

    values.forEach(value => {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        element.appendChild(option);
    }});

    if (values.includes(preferred)) {{
        element.value = preferred;
    }}
}}

fillSelect(
    "sourceFilter",
    metadata.sources,
    metadata.sources.includes("oanda") ? "oanda" : "all"
);
fillSelect("pairFilter", metadata.pairs);
fillSelect("sessionFilter", metadata.sessions);
fillSelect("versionFilter", metadata.versions);
fillSelect("timeframeFilter", metadata.timeframes);

function selected(id) {{
    return document.getElementById(id).value;
}}

function filteredTrades() {{
    return trades.filter(row =>
        (selected("sourceFilter") === "all"
            || row.data_source === selected("sourceFilter"))
        && (selected("pairFilter") === "all"
            || row.instrument === selected("pairFilter"))
        && (selected("sessionFilter") === "all"
            || row.session === selected("sessionFilter"))
        && (selected("versionFilter") === "all"
            || row.strategy_version === selected("versionFilter"))
        && (selected("timeframeFilter") === "all"
            || row.timeframe === selected("timeframeFilter"))
    );
}}

function number(value, digits = 3) {{
    if (value === null || value === undefined || !Number.isFinite(value)) {{
        return "—";
    }}
    return value.toFixed(digits);
}}

function summary(rows) {{
    const results = rows
        .map(row => Number(row.final_r))
        .filter(Number.isFinite);

    const trades = results.length;
    const net = results.reduce((sum, value) => sum + value, 0);
    const expectancy = trades ? net / trades : 0;
    const wins = results.filter(value => value > 0);
    const losses = results.filter(value => value < 0);
    const grossProfit = wins.reduce((sum, value) => sum + value, 0);
    const grossLoss = Math.abs(
        losses.reduce((sum, value) => sum + value, 0)
    );

    return {{
        trades,
        net,
        expectancy,
        winRate: trades ? wins.length / trades : 0,
        profitFactor: grossLoss > 0 ? grossProfit / grossLoss : null
    }};
}}

function confidenceMessage(count) {{
    if (count < 30) {{
        return "Very small sample. Do not use this result for live-trading decisions.";
    }}
    if (count < 100) {{
        return "Small sample. Treat this result as preliminary.";
    }}
    if (count < 300) {{
        return "Moderate sample. Additional out-of-sample testing is still required.";
    }}
    if (count < 1000) {{
        return "Good research sample, but robustness testing remains necessary.";
    }}
    return "Large sample. Continue checking regime stability and data quality.";
}}

function groupSummary(rows, key) {{
    const groups = new Map();

    rows.forEach(row => {{
        const value = row[key] ?? "unknown";
        if (!groups.has(value)) {{
            groups.set(value, []);
        }}
        groups.get(value).push(row);
    }});

    return [...groups.entries()]
        .map(([value, items]) => {{
            return {{
                group: value,
                ...summary(items)
            }};
        }})
        .sort((a, b) => b.expectancy - a.expectancy);
}}

function table(rows, firstLabel) {{
    if (!rows.length) {{
        return '<p class="empty">No matching data.</p>';
    }}

    const body = rows.map(row => `
        <tr>
            <td>${{row.group}}</td>
            <td>${{row.trades}}</td>
            <td>${{number(row.net)}}</td>
            <td>${{number(row.expectancy)}}</td>
            <td>${{(row.winRate * 100).toFixed(1)}}%</td>
            <td>${{number(row.profitFactor)}}</td>
        </tr>
    `).join("");

    return `
        <table>
            <thead>
                <tr>
                    <th>${{firstLabel}}</th>
                    <th>Trades</th>
                    <th>Net R</th>
                    <th>Expectancy R</th>
                    <th>Win rate</th>
                    <th>Profit factor</th>
                </tr>
            </thead>
            <tbody>${{body}}</tbody>
        </table>
    `;
}}

function update() {{
    const rows = filteredTrades();
    const overall = summary(rows);

    document.getElementById("tradeCount").textContent = overall.trades;
    document.getElementById("netR").textContent = number(overall.net);
    document.getElementById("expectancyR").textContent =
        number(overall.expectancy);
    document.getElementById("winRate").textContent =
        `${{(overall.winRate * 100).toFixed(1)}}%`;
    document.getElementById("profitFactor").textContent =
        number(overall.profitFactor);

    document.getElementById("sampleWarning").textContent =
        confidenceMessage(overall.trades);

    document.getElementById("pairTable").innerHTML =
        table(groupSummary(rows, "instrument"), "Pair");

    document.getElementById("sessionTable").innerHTML =
        table(groupSummary(rows, "session"), "Session");

    document.getElementById("weekdayTable").innerHTML =
        table(groupSummary(rows, "weekday"), "Weekday");

    document.getElementById("abTable").innerHTML =
        table(
            groupSummary(
                rows.map(row => ({{
                    ...row,
                    mtf_variant:
                        Number(row.h1_h4_agree) === 1
                        ? "H1/H4 agree"
                        : "H1/H4 disagree"
                }})),
                "mtf_variant"
            ),
            "Variant"
        );
}}

[
    "sourceFilter",
    "pairFilter",
    "sessionFilter",
    "versionFilter",
    "timeframeFilter"
].forEach(id => {{
    document.getElementById(id).addEventListener("change", update);
}});

document.getElementById("monteCarlo").textContent =
    Object.keys(monteCarlo).length
    ? JSON.stringify(monteCarlo, null, 2)
    : "Insufficient filtered real-data trades for Monte Carlo analysis.";

update();
</script>
</body>
</html>
"""

    output_path.write_text(
        document,
        encoding="utf-8",
    )
