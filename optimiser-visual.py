import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================

BASE_DIR = r"C:\Users\Wei Keong\Desktop\Y3S2\final-year-project\pytorch-neural-style-transfer-master\experiments (adam vs lbfgs)"

# label -> subfolder name under BASE_DIR
EXPERIMENT_DIRS = {
    "adam_vgg19":  os.path.join(BASE_DIR, "adam_vgg19"),
    "lbfgs_vgg16": os.path.join(BASE_DIR, "lbfgs_vgg16"),
    "lbfgs_vgg19": os.path.join(BASE_DIR, "lbfgs-vgg19"),
    "adam_vgg16":  os.path.join(BASE_DIR, "adam_vgg16"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "_comparison_output")


# =========================
# STEP 1: FIND + LOAD ALL summary.csv FILES
# =========================

def find_summary_csvs(folder):
    if not os.path.exists(folder):
        return []
    pattern = os.path.join(folder, "**", "summary.csv")
    return glob.glob(pattern, recursive=True)


def load_all_summaries():
    frames = []

    for label, folder in EXPERIMENT_DIRS.items():

        csv_paths = find_summary_csvs(folder)

        if not csv_paths:
            print(f"[SKIPPED] No summary.csv found for '{label}' under: {folder}")
            continue

        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[ERROR] Could not read {csv_path}: {e}")
                continue

            df["experiment_label"] = label
            df["source_file"] = csv_path
            frames.append(df)
            print(f"[LOADED] {label}: {csv_path}")

    if not frames:
        raise SystemExit(
            "\nNo summary.csv files were found in any of the experiment "
            "folders. Check EXPERIMENT_DIRS paths at the top of this script."
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined


# =========================
# STEP 2: BUILD COMPARISON TABLE
# =========================

def build_comparison_table(df):
    cols_needed = [
        "experiment_label", "optimizer", "model",
        "final_content_loss", "final_style_loss",
        "final_tv_loss", "time_taken"
    ]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing expected columns: {missing}")

    present_cols = [c for c in cols_needed if c in df.columns]
    summary = df[present_cols].groupby("experiment_label", as_index=False).mean(numeric_only=True)

    # re-attach optimizer/model labels (mean() drops non-numeric cols)
    meta = df[["experiment_label", "optimizer", "model"]].drop_duplicates("experiment_label")
    summary = summary.merge(meta, on="experiment_label", how="left", suffixes=("", "_meta"))

    return summary


def score_tradeoff(summary):
    s = summary.copy()

    for col in ["final_content_loss", "final_style_loss"]:
        lo, hi = s[col].min(), s[col].max()
        if hi > lo:
            s[col + "_norm"] = (s[col] - lo) / (hi - lo)
        else:
            s[col + "_norm"] = 0.0

    s["tradeoff_score"] = s["final_content_loss_norm"] + s["final_style_loss_norm"]
    return s.sort_values("tradeoff_score")


# =========================
# STEP 3: VISUALIZATIONS
# =========================

def plot_bar_comparison(summary, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = range(len(summary))
    width = 0.35

    ax.bar([i - width / 2 for i in x], summary["final_content_loss"],
           width, label="Content Loss")
    ax.bar([i + width / 2 for i in x], summary["final_style_loss"],
           width, label="Style Loss")

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["experiment_label"], rotation=20, ha="right")
    ax.set_ylabel("Loss")
    ax.set_title("Final Content vs Style Loss by Experiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def plot_tradeoff_scatter(summary, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(summary["final_content_loss"], summary["final_style_loss"], s=80)

    for _, row in summary.iterrows():
        ax.annotate(row["experiment_label"],
                    (row["final_content_loss"], row["final_style_loss"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("Final Content Loss (lower = closer to content)")
    ax.set_ylabel("Final Style Loss (lower = closer to style)")
    ax.set_title("Content vs Style Loss Trade-off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def plot_tradeoff_score_bar(scored, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(scored["experiment_label"], scored["tradeoff_score"], color="steelblue")
    ax.set_xlabel("Trade-off Score (lower = more balanced)")
    ax.set_title("Overall Content/Style Trade-off Ranking")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# =========================
# MAIN
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("==========================================")
    print("LOADING SUMMARY CSVs")
    print("==========================================")
    combined = load_all_summaries()

    print("\n==========================================")
    print("BUILDING COMPARISON TABLE")
    print("==========================================")
    summary = build_comparison_table(combined)
    print(summary.to_string(index=False))

    scored = score_tradeoff(summary)

    print("\n==========================================")
    print("TRADE-OFF RANKING (lower score = more balanced)")
    print("==========================================")
    print(scored[["experiment_label", "optimizer", "model",
                   "final_content_loss", "final_style_loss",
                   "tradeoff_score"]].to_string(index=False))

    best = scored.iloc[0]
    print(f"\nBest balanced trade-off: '{best['experiment_label']}' "
          f"(optimizer={best['optimizer']}, model={best['model']}, "
          f"score={best['tradeoff_score']:.4f})")

    print("\n==========================================")
    print("GENERATING VISUALIZATIONS")
    print("==========================================")
    plot_bar_comparison(summary, os.path.join(OUTPUT_DIR, "bar_comparison.png"))
    plot_tradeoff_scatter(summary, os.path.join(OUTPUT_DIR, "tradeoff_scatter.png"))
    plot_tradeoff_score_bar(scored, os.path.join(OUTPUT_DIR, "tradeoff_ranking.png"))

    # also save the combined + summary tables for the report
    combined_csv = os.path.join(OUTPUT_DIR, "combined_raw.csv")
    summary_csv = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    combined.to_csv(combined_csv, index=False)
    scored.to_csv(summary_csv, index=False)
    print(f"[SAVED] {combined_csv}")
    print(f"[SAVED] {summary_csv}")

    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
