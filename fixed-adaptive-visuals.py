import argparse
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HELPERS
# =============================================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def read_csv_row(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows[0]


def find_summary_files(root):
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    return sorted(root.rglob("summary.csv"))


def read_run(summary_path):
    summary_path = Path(summary_path)
    summary = read_csv_row(summary_path)

    run_dir = summary_path.parent
    iterations_path = run_dir / "iterations.csv"

    return {
        "summary_path": summary_path,
        "run_dir": run_dir,
        "iterations_path": iterations_path,
        "summary": summary,
    }


def is_adaptive(run):
    value = str(run["summary"].get("adaptive_weighting", "")).strip().lower()
    return value in {"yes", "true", "1", "adaptive"}


def run_key(run):
    s = run["summary"]

    return (
        str(s.get("content_image", "")).strip(),
        str(s.get("style_image", "")).strip(),
        str(s.get("model", "")).strip().lower(),
        str(s.get("optimizer", "")).strip().lower(),
        str(s.get("init_method", "")).strip().lower(),
    )


def describe_run(run):
    s = run["summary"]

    return (
        f'{s.get("model", "?").upper()} | '
        f'{s.get("optimizer", "?")} | '
        f'{s.get("content_image", "?")} + {s.get("style_image", "?")} | '
        f'{"ADAPTIVE" if is_adaptive(run) else "FIXED"}'
    )


def read_iterations(run):
    path = run["iterations_path"]

    if not path.exists():
        raise FileNotFoundError(f"iterations.csv not found: {path}")

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No iteration rows found: {path}")

    iterations = np.array(
        [safe_float(r.get("iteration")) for r in rows],
        dtype=float
    )

    total_loss = np.array(
        [safe_float(r.get("total_loss")) for r in rows],
        dtype=float
    )

    weighted_content = np.array(
        [safe_float(r.get("content_loss (weighted)")) for r in rows],
        dtype=float
    )

    weighted_style = np.array(
        [safe_float(r.get("style_loss (weighted)")) for r in rows],
        dtype=float
    )

    weighted_tv = np.array(
        [safe_float(r.get("tv_loss (weighted)")) for r in rows],
        dtype=float
    )

    # Adaptive runs contain these columns.
    content_weight = np.array(
        [safe_float(r.get("content_weight_used")) for r in rows],
        dtype=float
    )

    style_weight = np.array(
        [safe_float(r.get("style_weight_used")) for r in rows],
        dtype=float
    )

    # Fixed runs do not contain weight columns, so recover them from summary.
    if np.all(np.isnan(content_weight)):
        fixed_cw = safe_float(run["summary"].get("content_weight"))
        content_weight = np.full(len(rows), fixed_cw)

    if np.all(np.isnan(style_weight)):
        fixed_sw = safe_float(run["summary"].get("style_weight"))
        style_weight = np.full(len(rows), fixed_sw)

    return {
        "iteration": iterations,
        "total_loss": total_loss,
        "content": weighted_content,
        "style": weighted_style,
        "tv": weighted_tv,
        "content_weight": content_weight,
        "style_weight": style_weight,
    }


def normalized_two_way(content_value, style_value):
    values = np.array([content_value, style_value], dtype=float)

    if np.any(~np.isfinite(values)):
        return np.nan, np.nan

    total = values.sum()

    if total <= 0:
        return np.nan, np.nan

    return values[0] / total, values[1] / total


def final_contributions(run):
    data = read_iterations(run)

    content = data["content"][-1]
    style = data["style"][-1]

    if not np.isfinite(content):
        content = safe_float(run["summary"].get("final_content_loss"))

        cw = safe_float(run["summary"].get("content_weight"), 1.0)
        content *= cw

    if not np.isfinite(style):
        style = safe_float(run["summary"].get("final_style_loss"))

        sw = safe_float(run["summary"].get("style_weight"), 1.0)
        style *= sw

    return content, style


# =============================================================================
# MATCHING
# =============================================================================

def choose_latest(runs):
    if not runs:
        return None

    return max(
        runs,
        key=lambda r: r["summary_path"].stat().st_mtime
    )


def find_best_fixed_adaptive_pairs(root):
    summaries = find_summary_files(root)

    runs = [read_run(path) for path in summaries]

    fixed = {}
    adaptive = {}

    for run in runs:
        key = run_key(run)

        if is_adaptive(run):
            adaptive.setdefault(key, []).append(run)
        else:
            fixed.setdefault(key, []).append(run)

    pairs = []

    for key in sorted(set(fixed) & set(adaptive)):
        fixed_run = choose_latest(fixed[key])
        adaptive_run = choose_latest(adaptive[key])

        pairs.append((fixed_run, adaptive_run))

    return pairs

# =============================================================================
# FIGURE 1 — WEIGHT SCHEDULE
# =============================================================================

def plot_weight_comparison(fixed_run, adaptive_run, output_path):
    fixed_data = read_iterations(fixed_run)
    adaptive_data = read_iterations(adaptive_run)

    model = adaptive_run["summary"].get("model", "Model").upper()

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    # -------------------------------------------------------------------------
    # Content weight
    # -------------------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        adaptive_data["iteration"],
        adaptive_data["content_weight"],
        label="Adaptive content weight",
        linewidth=2
    )

    fixed_cw = safe_float(fixed_run["summary"].get("content_weight"))

    ax.axhline(
        fixed_cw,
        linestyle="--",
        label=f"Fixed content weight ({fixed_cw:.0f})"
    )

    ax.set_title(f"{model}: Content Weight — Fixed vs Adaptive")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Content Weight")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -------------------------------------------------------------------------
    # Style weight
    # -------------------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        adaptive_data["iteration"],
        adaptive_data["style_weight"],
        label="Adaptive style weight",
        linewidth=2
    )

    fixed_sw = safe_float(fixed_run["summary"].get("style_weight"))

    ax.axhline(
        fixed_sw,
        linestyle="--",
        label=f"Fixed style weight ({fixed_sw:.0f})"
    )

    ax.set_title(f"{model}: Style Weight — Fixed vs Adaptive")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Style Weight")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Fixed vs Adaptive Weighting",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# FIGURE 2 — FINAL CONTENT/STYLE TRADE-OFF
# =============================================================================

def plot_tradeoff_comparison(fixed_run, adaptive_run, output_path):
    fixed_content, fixed_style = final_contributions(fixed_run)
    adaptive_content, adaptive_style = final_contributions(adaptive_run)

    fc, fs = normalized_two_way(fixed_content, fixed_style)
    ac, ass = normalized_two_way(adaptive_content, adaptive_style)

    labels = ["Fixed", "Adaptive"]
    content_values = [fc, ac]
    style_values = [fs, ass]

    x = np.arange(len(labels))
    width = 0.65

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        x,
        content_values,
        width,
        label="Content contribution"
    )

    ax.bar(
        x,
        style_values,
        width,
        bottom=content_values,
        label="Style contribution"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalised Contribution")
    ax.set_xlabel("Weighting Method")
    ax.set_title("Final Content vs Style Contribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, value in enumerate(content_values):
        if np.isfinite(value):
            ax.text(
                i,
                value / 2,
                f"{value:.2f}",
                ha="center",
                va="center"
            )

    for i, (content_value, style_value) in enumerate(
        zip(content_values, style_values)
    ):
        if np.isfinite(style_value):
            ax.text(
                i,
                content_value + style_value / 2,
                f"{style_value:.2f}",
                ha="center",
                va="center"
            )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# FIGURE 3 — LOSS COMPARISON
# =============================================================================

def plot_loss_comparison(fixed_run, adaptive_run, output_path):
    fixed_data = read_iterations(fixed_run)
    adaptive_data = read_iterations(adaptive_run)

    model = adaptive_run["summary"].get("model", "Model").upper()

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    # Total loss
    ax = axes[0]

    ax.plot(
        fixed_data["iteration"],
        fixed_data["total_loss"],
        label="Fixed"
    )

    ax.plot(
        adaptive_data["iteration"],
        adaptive_data["total_loss"],
        label="Adaptive"
    )

    ax.set_title(f"{model}: Total Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Content/style contribution
    ax = axes[1]

    fixed_content = fixed_data["content"]
    fixed_style = fixed_data["style"]

    adaptive_content = adaptive_data["content"]
    adaptive_style = adaptive_data["style"]

    ax.plot(
        fixed_data["iteration"],
        fixed_content,
        linestyle="--",
        label="Fixed weighted content"
    )

    ax.plot(
        fixed_data["iteration"],
        fixed_style,
        linestyle="--",
        label="Fixed weighted style"
    )

    ax.plot(
        adaptive_data["iteration"],
        adaptive_content,
        label="Adaptive weighted content"
    )

    ax.plot(
        adaptive_data["iteration"],
        adaptive_style,
        label="Adaptive weighted style"
    )

    ax.set_title(f"{model}: Content/Style Loss Contributions")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Fixed vs Adaptive Optimisation Behaviour",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# CSV SUMMARY
# =============================================================================
def save_comparison_csv(pairs, output_path):
    rows = []

    for fixed_run, adaptive_run in pairs:
        fixed_content, fixed_style = final_contributions(fixed_run)
        adaptive_content, adaptive_style = final_contributions(adaptive_run)

        fixed_nc, fixed_ns = normalized_two_way(
            fixed_content,
            fixed_style
        )

        adaptive_nc, adaptive_ns = normalized_two_way(
            adaptive_content,
            adaptive_style
        )

        fs = fixed_run["summary"]
        ads = adaptive_run["summary"]

        rows.append({
            "content_image": ads.get("content_image", ""),
            "style_image": ads.get("style_image", ""),
            "model": ads.get("model", ""),
            "optimizer": ads.get("optimizer", ""),
            "init_method": ads.get("init_method", ""),

            "fixed_content_weight": fs.get("content_weight", ""),
            "fixed_style_weight": fs.get("style_weight", ""),

            "adaptive_final_content_weight": ads.get(
                "content_weight",
                ""
            ),
            "adaptive_final_style_weight": ads.get(
                "style_weight",
                ""
            ),

            "fixed_final_weighted_content": fixed_content,
            "fixed_final_weighted_style": fixed_style,

            "adaptive_final_weighted_content": adaptive_content,
            "adaptive_final_weighted_style": adaptive_style,

            "fixed_normalized_content": fixed_nc,
            "fixed_normalized_style": fixed_ns,

            "adaptive_normalized_content": adaptive_nc,
            "adaptive_normalized_style": adaptive_ns,

            "fixed_final_total_loss": fs.get("final_total_loss", ""),
            "adaptive_final_total_loss": ads.get("final_total_loss", ""),

            "fixed_time": fs.get("time_taken", ""),
            "adaptive_time": ads.get("time_taken", ""),

            "fixed_summary": str(fixed_run["summary_path"]),
            "adaptive_summary": str(adaptive_run["summary_path"]),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "content_image",
        "style_image",
        "model",
        "optimizer",
        "init_method",

        "fixed_content_weight",
        "fixed_style_weight",

        "adaptive_final_content_weight",
        "adaptive_final_style_weight",

        "fixed_final_weighted_content",
        "fixed_final_weighted_style",

        "adaptive_final_weighted_content",
        "adaptive_final_weighted_style",

        "fixed_normalized_content",
        "fixed_normalized_style",

        "adaptive_normalized_content",
        "adaptive_normalized_style",

        "fixed_final_total_loss",
        "adaptive_final_total_loss",

        "fixed_time",
        "adaptive_time",

        "fixed_summary",
        "adaptive_summary",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed-vs-adaptive NST visual comparisons."
    )

    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "data",
            "data-visual"
        ),
        help="Root directory containing NST experiment logs."
    )

    parser.add_argument(
        "--fixed",
        type=str,
        default=None,
        help="Path to a fixed run's summary.csv."
    )

    parser.add_argument(
        "--adaptive",
        type=str,
        default=None,
        help="Path to an adaptive run's summary.csv."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for generated figures and CSV."
    )

    args = parser.parse_args()

    root = Path(args.root)

    if args.output:
        output_root = Path(args.output)
    else:
        output_root = root / "fixed-adaptive-comparison"

    # -------------------------------------------------------------------------
    # Explicit pair
    # -------------------------------------------------------------------------
    if args.fixed and args.adaptive:

        fixed_run = read_run(Path(args.fixed))
        adaptive_run = read_run(Path(args.adaptive))

        pairs = [(fixed_run, adaptive_run)]

    # -------------------------------------------------------------------------
    # Automatic discovery
    # -------------------------------------------------------------------------
    else:

        pairs = find_best_fixed_adaptive_pairs(root)

        if not pairs:
            print()
            print("No matching fixed/adaptive experiment pair was found.")
            print()
            print("The two runs must have matching:")
            print("  - content image")
            print("  - style image")
            print("  - model")
            print("  - optimizer")
            print("  - init method")
            print()
            print("Example:")
            print("  python fixed-adaptive-visuals.py \\")
            print("      --fixed path/to/fixed/summary.csv \\")
            print("      --adaptive path/to/adaptive/summary.csv")
            return

    print()
    print("=" * 70)
    print("FIXED vs ADAPTIVE VISUAL ANALYSIS")
    print("=" * 70)

    all_pairs = []

    for index, (fixed_run, adaptive_run) in enumerate(pairs, start=1):

        model = adaptive_run["summary"].get("model", "model").lower()

        content = Path(
            adaptive_run["summary"].get("content_image", "content")
        ).stem

        style = Path(
            adaptive_run["summary"].get("style_image", "style")
        ).stem

        pair_name = f"{content}_{style}_{model}"

        pair_dir = output_root / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f"[PAIR {index}]")
        print(f"  Fixed    : {fixed_run['summary_path']}")
        print(f"  Adaptive : {adaptive_run['summary_path']}")

        # Figure 1
        weight_path = pair_dir / "01_fixed_vs_adaptive_weights.png"
        plot_weight_comparison(
            fixed_run,
            adaptive_run,
            weight_path
        )

        # Figure 2
        tradeoff_path = pair_dir / "02_fixed_vs_adaptive_tradeoff.png"
        plot_tradeoff_comparison(
            fixed_run,
            adaptive_run,
            tradeoff_path
        )

        # Figure 3
        loss_path = pair_dir / "03_fixed_vs_adaptive_loss_comparison.png"
        plot_loss_comparison(
            fixed_run,
            adaptive_run,
            loss_path
        )

        all_pairs.append((fixed_run, adaptive_run))

        print(f"  [OK] {weight_path}")
        print(f"  [OK] {tradeoff_path}")
        print(f"  [OK] {loss_path}")

    comparison_csv = output_root / "fixed_vs_adaptive_comparison.csv"

    save_comparison_csv(
        all_pairs,
        comparison_csv
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output directory : {output_root}")
    print(f"Comparison CSV   : {comparison_csv}")
    print()
    print("Generated figures:")
    print("  01_fixed_vs_adaptive_weights.png")
    print("  02_fixed_vs_adaptive_tradeoff.png")
    print("  03_fixed_vs_adaptive_loss_comparison.png")


if __name__ == "__main__":
    main()