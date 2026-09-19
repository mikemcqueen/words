"""Plot the frequency distribution of probability values."""

from __future__ import annotations

import argparse

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot_probabilities(source: Path, output: Path) -> int:
    """Plot values from source to output and return the number of values."""
    counts = Counter(map(float, source.read_text().split()))
    if not counts:
        raise SystemExit(f"no probability values in {source}")

    values = sorted(counts)
    frequencies = [counts[value] for value in values]
    total = sum(frequencies)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(figsize=(12, 7.5), dpi=160)
    bars = axes.bar(
        [f"{value:.2f}".removeprefix("0") for value in values],
        frequencies,
        color="#4472C4",
        edgecolor="#244A82",
        linewidth=0.8,
    )
    axes.set_title(f"Distribution of values in {source}", fontsize=16, pad=14)
    axes.set_xlabel("Value", fontsize=12)
    axes.set_ylabel("Count", fontsize=12)
    axes.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    axes.tick_params(axis="x", labelsize=8)
    axes.grid(axis="x", visible=False)
    axes.set_axisbelow(True)
    axes.margins(x=0.025)
    axes.set_ylim(0, max(frequencies) * 1.12)

    for bar, frequency in zip(bars, frequencies):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            frequency + max(frequencies) * 0.012,
            f"{frequency:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    figure.text(
        0.99,
        0.01,
        f"n = {total:,}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    figure.savefig(
        output,
        metadata={
            "Software": "Matplotlib",
            "Title": f"{source.name} value distribution",
        },
    )
    plt.close(figure)
    return total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the frequency distribution of probability values."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="input file containing probability values",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output PNG (default: INPUT.png)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output = args.output or Path(f"{args.input}.png")
    total = plot_probabilities(args.input, output)
    print(f"plotted {total:,} values to {output}")


if __name__ == "__main__":
    main()
