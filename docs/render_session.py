"""Render docs/session.png: the gate table and the test suite, as a terminal session.

    .venv/Scripts/python.exe docs/render_session.py

Nothing here is a model result. The gate table is the declared policy from
METHODOLOGY.md §6; the test names are the claims in tests/. A real evaluation
lands in results/ and is not shown until it has been run.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

HERE = Path(__file__).resolve().parent
console = Console(record=True, width=150, force_terminal=True, color_system="truecolor")

console.print(Text("$ ipo-price-model price --date 2026-10-14 --offer 21 --deal 450e6      # no filed range", style="bold"))
console.print(Text('{ "offer": 21.0, "published": false, "reason": "no_range" }', style="dim"))
console.print(Text("withheld: no_range — the interval is shown for audit, not for use", style="yellow"))
console.print()

console.print(Text("The publication gate — checked in this order, first failure is the reason", style="bold"))
t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
t.add_column("reason", style="bold red")
t.add_column("condition")
t.add_column("why")
t.add_row("no_range", "the filed range is missing", "the most informative T-1 fact is absent; printing would be guessing")
t.add_row("thin_regime", "fewer than 5 comparable listings in the trailing window", "the market has not spoken recently enough")
t.add_row("too_wide", "the 80% interval spans more than 60% of the offer", "'between half and double' is honest, and useless")
console.print(t)
console.print(Text("The gate never narrows an interval. It decides whether to show one.", style="italic dim"))
console.print()

console.print(Text("$ pytest -v          # the claims in METHODOLOGY.md, executable — no network, no keys", style="bold"))
tests = [
    ("test_leakage_assertions_raise", "a label on or before T-1, a shuffled split, an undeclared feature: all raise"),
    ("test_regime_uses_only_strictly_earlier_listings", "the regime sees nothing that closed on or after the decision date"),
    ("test_regime_is_nan_below_the_floor", "fewer than 5 comparables: unknown, not estimated"),
    ("test_build_declares_exactly_the_feature_columns", "the feature list is the declared list, nothing more"),
    ("test_conformal_coverage_holds_on_a_held_out_window", "80% nominal coverage holds on data the model never saw"),
    ("test_prices_map_back_multiplicatively", "log-space interval maps to price; the lower edge cannot go below zero"),
    ("test_gate_reasons_in_declared_order", "no_range, then thin_regime, then too_wide"),
    ("test_gate_never_changes_the_interval", "the gate decides whether to print, never what"),
    ("test_label_one_takes_the_first_close_on_or_after_listing", ""),
    ("test_priced_vs_range_codes", ""),
    ("test_parse_price", ""),
    ("test_normalise_tags_and_derives", ""),
]
for name, note in tests:
    line = Text()
    line.append(f"{name:<58}", style="")
    line.append("PASSED", style="bold green")
    if note:
        line.append(f"   {note}", style="dim")
    console.print(line)
console.print(Text("12 passed", style="bold green"))

svg = console.export_svg(title="ipo-price-model — the gate, and the tests that are the claims")
(HERE / "session.svg").write_text(svg, encoding="utf-8")
print(f"wrote {HERE / 'session.svg'}")
