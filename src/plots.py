"""Publication-quality figures for the gemini-reasoning-eval benchmark.

Generates 4 figures matching the paper's visual style:
    1. Token breakdown (stacked bar, 4 components: image, text, thought, response)
    2. Budget scaling (dual line chart by mean reasoning tokens)
    3. Quality tiers (perfect F1 rate + failure rate)
    4. Family comparison (Flash vs Lite)
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

logger = logging.getLogger(__name__)

# Token component colors (matching paper Figure 1)
TOK_IMAGE = "#4381C1"    # Image Tokens (input, blue)
TOK_TEXT = "#7B8CDE"     # Text Tokens (input, purple)
TOK_THOUGHT = "#EF8354"  # Thought Tokens (orange)
TOK_RESP = "#43AA8B"     # Response Tokens (green)

# Family line colors
COLOR_FLASH = "#4381C1"
COLOR_LITE = "#EF8354"


def _get_variant_label(model: str, budget: int) -> str:
    """Generate a human-readable label matching paper format.

    Examples:
        gemini-2.5-flash, 128   -> "Flash – 128"
        gemini-2.5-flash, -1    -> "Flash – Dynamic"
        gemini-2.5-flash-lite, 512  -> "Lite – 512"
    """
    short = "Lite" if "flash-lite" in model else "Flash"
    if budget == -1:
        return f"{short} – Dynamic"
    return f"{short} – {budget}"


def _get_variant_color(variants: list[dict], idx: int) -> str:
    """Assign colors based on variant index."""
    palette = ["#EF8354", "#43AA8B", "#2D6A4F", "#4381C1", "#7B8CDE", "#D64045"]
    return palette[idx % len(palette)]


def _save_fig(fig: plt.Figure, name: str, output_dir: str, fmt: str, dpi: int) -> None:
    path = Path(output_dir) / f"{name}.{fmt}"
    fig.savefig(path, format=fmt, dpi=dpi)
    plt.close(fig)
    logger.info(f"  Saved {path.name}")


def plot_token_breakdown(
    variants: list[dict], output_dir: str, fmt: str = "png", dpi: int = 300,
) -> None:
    """Stacked bar chart with 4 token components per variant (matching paper Figure 1).

    Components: Image Tokens (input), Text Tokens (input), Thought Tokens, Response Tokens.
    """
    thinking = [v for v in variants if v["thinking_budget"] != 0]
    if not thinking:
        logger.warning("No thinking variants found for token breakdown plot.")
        return

    labels = [_get_variant_label(v["model"], v["thinking_budget"]) for v in thinking]

    image_tokens = [v.get("image_tokens_mean", 0) for v in thinking]
    text_tokens = [v.get("text_tokens_mean", 0) for v in thinking]
    thought_tokens = [v.get("thought_tokens_mean", 0) for v in thinking]
    output_tokens = [v.get("output_tokens_mean", 0) for v in thinking]
    total_tokens = [v.get("total_tokens_mean", 0) for v in thinking]

    # Fallback: if image/text breakdown not available, derive input = total - thought - output
    has_breakdown = any(i > 0 for i in image_tokens) or any(t > 0 for t in text_tokens)
    if not has_breakdown:
        # Try stored input_tokens_mean first; if zero derive from total
        stored_input = [v.get("input_tokens_mean", 0) for v in thinking]
        input_tokens = [
            s if s > 0 else max(0, tot - tho - out)
            for s, tot, tho, out in zip(stored_input, total_tokens, thought_tokens, output_tokens)
        ]
        components = [
            (np.array(input_tokens), "Input Tokens", TOK_IMAGE),
            (np.array(thought_tokens), "Thought Tokens", TOK_THOUGHT),
            (np.array(output_tokens), "Response Tokens", TOK_RESP),
        ]
        totals = np.array(input_tokens) + np.array(thought_tokens) + np.array(output_tokens)
        if not any(t > 0 for t in totals):
            totals = np.array(total_tokens)
    else:
        components = [
            (np.array(image_tokens), "Image Tokens", TOK_IMAGE),
            (np.array(text_tokens), "Text Tokens", TOK_TEXT),
            (np.array(thought_tokens), "Thought Tokens", TOK_THOUGHT),
            (np.array(output_tokens), "Response Tokens", TOK_RESP),
        ]
        totals = (
            np.array(image_tokens)
            + np.array(text_tokens)
            + np.array(thought_tokens)
            + np.array(output_tokens)
        )
        # Use API total if per-component sum is zero (data not yet available)
        if not any(t > 0 for t in totals):
            totals = np.array(total_tokens)

    x = np.arange(len(labels))
    w = 0.55

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bottom = np.zeros(len(labels))
    for vals, label, color in components:
        ax.bar(x, vals, w, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if totals[i] > 0 and v / totals[i] > 0.06:
                ax.text(i, b + v / 2, f"{v:,.0f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals

    for i, t in enumerate(totals):
        ax.text(i, t + t * 0.02, f"{t:,.0f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Tokens")
    ax.set_title("Mean Token Breakdown per Scene (from Gemini API)", fontweight="bold", pad=10)
    ax.set_ylim(0, max(float(totals.max()) * 1.12, 1.0) if len(totals) else 1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=9, loc="upper left", frameon=True,
              edgecolor="#DDDDDD", fancybox=False)

    plt.tight_layout()
    _save_fig(fig, "fig01_token_breakdown", output_dir, fmt, dpi)


def plot_budget_scaling(
    variants: list[dict], output_dir: str, fmt: str = "png", dpi: int = 300,
) -> None:
    """Dual line chart: metrics vs. mean reasoning tokens, separate series per family.

    Left panel: Contentfulness and F1.
    Right panel: Recall (Thought Coverage) and Precision (Output Grounding).
    X-axis: mean thought tokens (continuous).
    """
    thinking = [v for v in variants if v["thinking_budget"] != 0]
    if not thinking:
        return

    # Group into Flash and Lite families
    flash_variants = sorted(
        [v for v in thinking if "flash-lite" not in v["model"]],
        key=lambda v: v.get("thought_tokens_mean", 0),
    )
    lite_variants = sorted(
        [v for v in thinking if "flash-lite" in v["model"]],
        key=lambda v: v.get("thought_tokens_mean", 0),
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5),
                                      gridspec_kw={"wspace": 0.28})

    def plot_family_lines(ax: plt.Axes, family_variants: list[dict],
                          metrics: list[str], labels: list[str],
                          colors: list[str], linestyles: list[str],
                          family_label: str) -> None:
        if not family_variants:
            return
        xs = [v.get("thought_tokens_mean", 0) for v in family_variants]
        for metric, label, color, ls in zip(metrics, labels, colors, linestyles):
            ys = [v.get(metric) or 0 for v in family_variants]
            ax.plot(xs, ys, marker="o", color=color, linestyle=ls,
                    linewidth=2, markersize=7,
                    label=f"{family_label} – {label}")
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8.5)

    # Left panel: Contentfulness & F1
    left_metrics = ["contentfulness_mean", "f1_mean"]
    left_labels = ["Contentfulness", "F1"]
    flash_colors_l = ["#4381C1", "#D64045"]
    lite_colors_l = ["#4381C1", "#D64045"]

    plot_family_lines(ax_l, flash_variants, left_metrics, left_labels,
                      flash_colors_l, ["-", "-"], "Flash")
    plot_family_lines(ax_l, lite_variants, left_metrics, left_labels,
                      lite_colors_l, ["--", "--"], "Lite")

    ax_l.set_xlabel("Mean Reasoning Tokens")
    ax_l.set_ylim(0, 1.08)
    ax_l.set_ylabel("Score")
    ax_l.set_title("Contentfulness & F1", fontweight="bold", pad=10)
    ax_l.legend(fontsize=9, frameon=True, edgecolor="#DDDDDD", fancybox=False)

    # Right panel: Recall (TC) & Precision (OG)
    right_metrics = ["thought_coverage_mean", "output_grounding_mean"]
    right_labels = ["Recall", "Precision"]
    flash_colors_r = ["#EF8354", "#43AA8B"]
    lite_colors_r = ["#EF8354", "#43AA8B"]

    plot_family_lines(ax_r, flash_variants, right_metrics, right_labels,
                      flash_colors_r, ["-", "-"], "Flash")
    plot_family_lines(ax_r, lite_variants, right_metrics, right_labels,
                      lite_colors_r, ["--", "--"], "Lite")

    ax_r.set_xlabel("Mean Reasoning Tokens")
    ax_r.set_ylim(0, 1.08)
    ax_r.set_ylabel("Score")
    ax_r.set_title("Recall & Precision", fontweight="bold", pad=10)
    ax_r.legend(fontsize=9, frameon=True, edgecolor="#DDDDDD", fancybox=False)

    plt.tight_layout()
    _save_fig(fig, "fig02_budget_scaling", output_dir, fmt, dpi)


def plot_quality_tiers(
    variants: list[dict], output_dir: str, fmt: str = "png", dpi: int = 300,
) -> None:
    """Two-panel: Perfect F1 rate and Failure rate per variant."""
    thinking = [v for v in variants if v["thinking_budget"] != 0]
    if not thinking:
        return

    labels = [_get_variant_label(v["model"], v["thinking_budget"]) for v in thinking]
    colors = [_get_variant_color(thinking, i) for i in range(len(thinking))]
    perfect = [v.get("perfect_f1_pct") or 0 for v in thinking]
    low = [v.get("low_f1_pct") or 0 for v in thinking]

    x = np.arange(len(labels))
    w = 0.55

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5),
                                      gridspec_kw={"wspace": 0.30})

    # Left: Perfect F1
    bars = ax_l.bar(x, perfect, w, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, perfect):
        ax_l.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                  f"{val:.1f}%", ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold", color="#333333")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=10)
    ax_l.set_ylabel("% of Scenes")
    ax_l.set_title("Perfect Score Rate (F1 = 1.0)", fontweight="bold", fontsize=12, pad=10)
    ax_l.set_ylim(0, max(max(perfect) * 1.18, 1.0) if perfect else 1)

    # Right: Low F1
    bars = ax_r.bar(x, low, w, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, low):
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                  f"{val:.1f}%", ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold", color="#333333")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, fontsize=10)
    ax_r.set_ylabel("% of Scenes")
    ax_r.set_title("Failure Rate (F1 < 0.5)", fontweight="bold", fontsize=12, pad=10)
    ax_r.set_ylim(0, max(low) * 1.25 if low and max(low) > 0 else 1)

    plt.tight_layout()
    _save_fig(fig, "fig03_quality_tiers", output_dir, fmt, dpi)


def plot_family_comparison(
    variants: list[dict], output_dir: str, fmt: str = "png", dpi: int = 300,
) -> None:
    """Grouped bar chart comparing Flash vs Lite families across 4 metrics.

    Metric labels use paper terminology: Recall (Thought Coverage), Precision (Output Grounding).
    """
    thinking = [v for v in variants if v["thinking_budget"] != 0]
    if not thinking:
        return

    # Group by model family
    families: dict[str, list[dict]] = {}
    for v in thinking:
        family = v["model"]
        if family not in families:
            families[family] = []
        families[family].append(v)

    if len(families) < 1:
        return

    metrics = ["contentfulness_mean", "thought_coverage_mean", "output_grounding_mean", "f1_mean"]
    # Use paper terminology: Recall = Thought Coverage, Precision = Output Grounding
    metric_names = ["Contentfulness", "Recall", "Precision", "F1"]

    n_families = len(families)
    fig, axes = plt.subplots(1, n_families, figsize=(6.5 * n_families, 5),
                              gridspec_kw={"wspace": 0.28})

    if n_families == 1:
        axes = [axes]

    palette = ["#EF8354", "#4381C1", "#43AA8B", "#2D6A4F", "#7B8CDE"]

    for ax, (family_name, family_variants) in zip(axes, families.items()):
        x = np.arange(len(metrics))
        w = 0.35

        for i, v in enumerate(family_variants):
            vals = [v.get(m) or 0 for m in metrics]
            off = (i - (len(family_variants) - 1) / 2) * w
            label = _get_variant_label(v["model"], v["thinking_budget"])
            color = palette[i % len(palette)]
            bars = ax.bar(x + off, vals, width=w * 0.88, color=color,
                          label=label, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8.5, color="#333333")

        # Short family title: "Gemini 2.5 Flash" or "Gemini 2.5 Flash Lite"
        short_family = (
            family_name
            .replace("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite")
            .replace("gemini-2.5-flash", "Gemini 2.5 Flash")
        )
        ax.set_title(short_family, fontweight="bold", fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Mean Score")
        ax.legend(fontsize=9.5, frameon=True, edgecolor="#DDDDDD",
                  fancybox=False, loc="upper left")

    plt.tight_layout()
    _save_fig(fig, "fig04_family_comparison", output_dir, fmt, dpi)


def generate_figures(
    aggregated: dict[str, Any],
    output_dir: str,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """Generate all publication figures from aggregated results."""
    variants = aggregated.get("variants", [])
    if not variants:
        logger.warning("No variant data found in aggregated results.")
        return

    logger.info(f"Generating figures for {len(variants)} variants...")

    plot_token_breakdown(variants, output_dir, fmt, dpi)
    plot_budget_scaling(variants, output_dir, fmt, dpi)
    plot_quality_tiers(variants, output_dir, fmt, dpi)
    plot_family_comparison(variants, output_dir, fmt, dpi)

    logger.info("All figures generated.")
