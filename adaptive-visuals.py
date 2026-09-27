import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================

SEARCH_ROOTS = [
    "experiments (adaptive_weighting)",
    os.path.join("data", "data-visual"),
]

OUTPUT_DIR = "experiments (adaptive_weighting)/_plots"

# =========================
# STEP 1: FIND iterations.csv FOR EACH MODEL
# =========================

def _find_iteration_csvs(roots):
    found = []

    for root in roots:
        if not os.path.exists(root):
            continue

        pattern = os.path.join(root, "**", "iterations.csv")
        found.extend(glob.glob(pattern, recursive=True))

    return found


def find_adaptive_iterations_csv(model_filter, roots):
    all_csvs = _find_iteration_csvs(roots)

    candidates = []

    for path in all_csvs:

        if model_filter.lower() not in path.lower():
            continue

        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception as e:
            print(f"[WARNING] Could not read header of {path}: {e}")
            continue

        if "content_weight_used" not in header:
            # Not an adaptive-weighting run — skip it.
            continue

        candidates.append((path, os.path.getmtime(path)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


VGG16_CSV = find_adaptive_iterations_csv("vgg16", SEARCH_ROOTS)
VGG19_CSV = find_adaptive_iterations_csv("vgg19", SEARCH_ROOTS)


# =========================
# STEP 2: LOAD DATA
# =========================

def load_iterations(csv_path, label):

    if csv_path is None:
        print(f"[SKIPPED] No adaptive-weighting iterations.csv found for {label}.")
        return None

    print(f"[LOADED] {label}: {csv_path}")

    df = pd.read_csv(csv_path)

    df["iteration"] = df["iteration"].astype(int)

    return df


vgg16_df = load_iterations(VGG16_CSV, "vgg16")
vgg19_df = load_iterations(VGG19_CSV, "vgg19")

if vgg16_df is None and vgg19_df is None:
    raise SystemExit(
        "\nNo adaptive-weighting iterations.csv files were found for "
        "either model. Check SEARCH_ROOTS at the top of this script, "
        "and confirm both runs were generated with --adaptive_weighting."
    )

# =========================
# STEP 3: PLOTTING HELPER
# =========================

def plot_metric_comparison(vgg16_df, vgg19_df, column, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted_anything = False

    if vgg16_df is not None and column in vgg16_df.columns:
        ax.plot(
            vgg16_df["iteration"], vgg16_df[column],
            label="VGG16", color="tab:green"
        )
        plotted_anything = True

    if vgg19_df is not None and column in vgg19_df.columns:
        ax.plot(
            vgg19_df["iteration"], vgg19_df[column],
            label="VGG19", color="tab:purple"
        )
        plotted_anything = True

    if not plotted_anything:
        print(f"[SKIPPED] Column '{column}' not found in either run — skipping {title}.")
        plt.close(fig)
        return

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# =========================
# STEP 4: GENERATE THE 5 REQUESTED GRAPHS
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n==========================================")
print("GENERATING ADAPTIVE VGG16 vs VGG19 GRAPHS")
print("==========================================")

plot_metric_comparison(
    vgg16_df, vgg19_df,
    column="content_loss (weighted)",
    title="Content Loss vs Iteration (Adaptive Weighting)",
    ylabel="Content Loss (weighted)",
    out_path=os.path.join(OUTPUT_DIR, "01_content_loss_vs_iteration.png")
)

plot_metric_comparison(
    vgg16_df, vgg19_df,
    column="style_loss (weighted)",
    title="Style Loss vs Iteration (Adaptive Weighting)",
    ylabel="Style Loss (weighted)",
    out_path=os.path.join(OUTPUT_DIR, "02_style_loss_vs_iteration.png")
)

plot_metric_comparison(
    vgg16_df, vgg19_df,
    column="total_loss",
    title="Total Loss vs Iteration (Adaptive Weighting)",
    ylabel="Total Loss",
    out_path=os.path.join(OUTPUT_DIR, "03_total_loss_vs_iteration.png")
)

plot_metric_comparison(
    vgg16_df, vgg19_df,
    column="content_weight_used",
    title="Content Weight vs Iteration (Adaptive Weighting)",
    ylabel="Content Weight",
    out_path=os.path.join(OUTPUT_DIR, "04_content_weight_vs_iteration.png")
)

plot_metric_comparison(
    vgg16_df, vgg19_df,
    column="style_weight_used",
    title="Style Weight vs Iteration (Adaptive Weighting)",
    ylabel="Style Weight",
    out_path=os.path.join(OUTPUT_DIR, "05_style_weight_vs_iteration.png")
)

print(f"\nAll plots saved to: {OUTPUT_DIR}")