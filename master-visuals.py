import os
import re
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# SETTINGS
# ==========================================================

BASE_DIR = (
    r"C:\Users\Wei Keong\Desktop\Y3S2\final-year-project"
    r"\pytorch-neural-style-transfer-master"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "_master_comparison"
)


# ==========================================================
# EXPERIMENT METADATA
# ==========================================================

EXPERIMENTS = [

    {
        "group": "Optimizer x Model",
        "order": 1,
        "label": "Adam — VGG16",
        "path": os.path.join(
            BASE_DIR,
            "data", "data-visual",
            "mount-fuji_ancient-chinese-style_vgg16",
            "mount-fuji_ancient-chinese-style_vgg16_test_002"
        ),
        "optimizer": "Adam",
        "model": "VGG16",
        "content_weight": None,
        "adaptive": False,
    },

    {
        "group": "Optimizer x Model",
        "order": 2,
        "label": "Adam — VGG19",
        "path": os.path.join(
            BASE_DIR,
            "data", "data-visual",
            "mount-fuji_ancient-chinese-style_vgg19",
            "mount-fuji_ancient-chinese-style_vgg19_test_002"
        ),
        "optimizer": "Adam",
        "model": "VGG19",
        "content_weight": None,
        "adaptive": False,
    },

    {
        "group": "Optimizer x Model",
        "order": 3,
        "label": "L-BFGS — VGG16",
        "path": os.path.join(
            BASE_DIR,
            "experiments (adam vs lbfgs)",
            "lbfgs_vgg16"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG16",
        "content_weight": None,
        "adaptive": False,
    },

    {
        "group": "Optimizer x Model",
        "order": 4,
        "label": "L-BFGS — VGG19",
        "path": os.path.join(
            BASE_DIR,
            "experiments (adam vs lbfgs)",
            "lbfgs-vgg19"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG19",
        "content_weight": None,
        "adaptive": False,
    },


    # ======================================================
    # 2. ADAPTIVE WEIGHTING
    # ======================================================

    {
        "group": "Adaptive Weighting",
        "order": 1,
        "label": "Adaptive — L-BFGS VGG16",
        "path": os.path.join(
            BASE_DIR,
            "experiments (adaptive_weighting)",
            "adaptive_test_vgg16_lbfgs"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG16",
        "content_weight": None,
        "adaptive": True,
    },

    {
        "group": "Adaptive Weighting",
        "order": 2,
        "label": "Adaptive — L-BFGS VGG19",
        "path": os.path.join(
            BASE_DIR,
            "experiments (adaptive_weighting)",
            "adaptive_test_vgg19_lbfgs"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG19",
        "content_weight": None,
        "adaptive": True,
    },


    # ======================================================
    # 3. CONTENT-WEIGHT SWEEP
    # ======================================================

    {
        "group": "Content-Weight Sweep",
        "order": 1,
        "label": "CW = 50,000",
        "path": os.path.join(
            BASE_DIR,
            "experiments (content_weight_vgg16_LBFGS)"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG16",
        "content_weight": 50000,
        "adaptive": False,
        "search_token": "50000",
    },

    {
        "group": "Content-Weight Sweep",
        "order": 2,
        "label": "CW = 75,000",
        "path": os.path.join(
            BASE_DIR,
            "experiments (content_weight_vgg16_LBFGS)"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG16",
        "content_weight": 75000,
        "adaptive": False,
        "search_token": "75000",
    },

    {
        "group": "Content-Weight Sweep",
        "order": 3,
        "label": "CW = 100,000 (Baseline)",
        "path": os.path.join(
            BASE_DIR,
            "experiments (content_weight_vgg16_LBFGS)"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG16",
        "content_weight": 100000,
        "adaptive": False,
        "search_token": "100000",
    },

    {
        "group": "Content-Weight Sweep",
        "order": 4,
        "label": "CW = 150,000",
        "path": os.path.join(
            BASE_DIR,
            "experiments (content_weight_vgg16_LBFGS)"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG16",
        "content_weight": 150000,
        "adaptive": False,
        "search_token": "150000",
    },

    {
        "group": "Content-Weight Sweep",
        "order": 5,
        "label": "CW = 200,000",
        "path": os.path.join(
            BASE_DIR,
            "experiments (content_weight_vgg16_LBFGS)"
        ),
        "optimizer": "L-BFGS",
        "model": "VGG16",
        "content_weight": 200000,
        "adaptive": False,
        "search_token": "200000",
    },
]

# ==========================================================
# FIND iterations.csv
# ==========================================================

def find_iterations_csv(folder_path, search_token=None):

    if not os.path.exists(folder_path):
        return None

    matches = glob.glob(
        os.path.join(folder_path, "**", "iterations.csv"),
        recursive=True
    )

    if not matches:
        return None

    if search_token is not None:
        pattern = re.compile(rf"(?<!\d){re.escape(search_token)}(?!\d)")

        matches = [
            path for path in matches
            if pattern.search(path.replace("\\", "/"))
        ]

        if not matches:
            return None
        def parent_is_exact(path):
            parent_name = os.path.basename(os.path.dirname(path))
            return bool(pattern.search(parent_name))

        exact_parent_matches = [p for p in matches if parent_is_exact(p)]
        if exact_parent_matches:
            matches = exact_parent_matches

    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


# ==========================================================
# SUMMARIZE ONE RUN
# ==========================================================

def summarize_run(csv_path):

    df = pd.read_csv(csv_path)

    if "iteration" not in df.columns:
        raise ValueError(
            f"'iteration' column not found in {csv_path}"
        )

    df["iteration"] = df["iteration"].astype(int)


    summary = {

        "iterations_logged": len(df),
        "final_content_loss": None,
        "final_style_loss": None,
        "final_total_loss": None,
        "best_total_loss": None,
        "best_iteration": None,
        "final_content_weight_used": None,
        "final_style_weight_used": None,
    }

    # ======================================================
    # FINAL CONTENT LOSS
    # ======================================================

    if "content_loss (weighted)" in df.columns:

        summary["final_content_loss"] = (
            df["content_loss (weighted)"].iloc[-1]
        )

    # ======================================================
    # FINAL STYLE LOSS
    # ======================================================

    if "style_loss (weighted)" in df.columns:

        summary["final_style_loss"] = (
            df["style_loss (weighted)"].iloc[-1]
        )

    # ======================================================
    # TOTAL LOSS
    # ======================================================

    if "total_loss" in df.columns:

        summary["final_total_loss"] = (
            df["total_loss"].iloc[-1]
        )

        best_idx = df["total_loss"].idxmin()

        summary["best_total_loss"] = (
            df["total_loss"].loc[best_idx]
        )

        summary["best_iteration"] = int(
            df["iteration"].loc[best_idx]
        )

    # =====================================================
    # ADAPTIVE WEIGHTS
    # =====================================================

    if "content_weight_used" in df.columns:

        summary["final_content_weight_used"] = (
            df["content_weight_used"].iloc[-1]
        )

    if "style_weight_used" in df.columns:

        summary["final_style_weight_used"] = (
            df["style_weight_used"].iloc[-1]
        )

    return summary

# ==========================================================
# BUILD MASTER TABLE
# ==========================================================

def build_master_table(experiments):

    rows = []


    for exp in experiments:

        search_token = exp.get(
            "search_token",
            None
        )

        csv_path = find_iterations_csv(
            exp["path"],
            search_token
        )

        # ==================================================
        # BASE METADATA
        # ==================================================

        row = {

            "Group":
                exp["group"],

            "Order":
                exp["order"],

            "Experiment":
                exp["label"],

            "Optimizer":
                exp["optimizer"],

            "Model":
                exp["model"],

            "Content Weight":
                exp["content_weight"],

            "Adaptive":
                "Yes"
                if exp["adaptive"]
                else "No",
        }


        # ==================================================
        # MISSING
        # ==================================================

        if csv_path is None:

            print(
                f"[MISSING] "
                f"{exp['label']}"
            )

            print(
                f"          Search path: "
                f"{exp['path']}"
            )

            if search_token:

                print(
                    f"          Search token: "
                    f"{search_token}"
                )


            row.update({

                "Iterations Logged":
                    None,

                "Final Content Loss":
                    None,

                "Final Style Loss":
                    None,

                "Final Total Loss":
                    None,

                "Best Total Loss":
                    None,

                "Best Iteration":
                    None,

                "Final Content Weight":
                    None,

                "Final Style Weight":
                    None,

                "Source":
                    "NOT FOUND",
            })


        # ==================================================
        # FOUND
        # ==================================================

        else:

            print(
                f"[FOUND]   "
                f"{exp['label']}"
            )

            print(
                f"          {csv_path}"
            )


            s = summarize_run(
                csv_path
            )


            row.update({

                "Iterations Logged":
                    s["iterations_logged"],

                "Final Content Loss":
                    s["final_content_loss"],

                "Final Style Loss":
                    s["final_style_loss"],

                "Final Total Loss":
                    s["final_total_loss"],

                "Best Total Loss":
                    s["best_total_loss"],

                "Best Iteration":
                    s["best_iteration"],

                "Source":
                    os.path.relpath(
                        csv_path,
                        BASE_DIR
                    ),
            })

            if exp["adaptive"]:

                row["Final Content Weight"] = (
                    s["final_content_weight_used"]
                )

                row["Final Style Weight"] = (
                    s["final_style_weight_used"]
                )

            else:

                row["Final Content Weight"] = None

                row["Final Style Weight"] = None


        rows.append(row)


    master_df = pd.DataFrame(
        rows
    )

    # ==========================================================
    # SORTING
    # ==========================================================

    group_order = [

        "Optimizer x Model",
        "Adaptive Weighting",
        "Content-Weight Sweep",
    ]


    master_df["Group"] = pd.Categorical(

        master_df["Group"],
        categories=group_order,
        ordered=True
    )


    master_df = (
        master_df
        .sort_values(
            ["Group", "Order"]
        )
        .reset_index(drop=True)
    )
    # Order column is only for internal sorting
    master_df = master_df.drop(
        columns=["Order"]
    )
    return master_df

# ==========================================================
# NUMBER FORMATTING
# ==========================================================

def fmt_num(x):

    if x is None:
        return "-"
    if pd.isna(x):
        return "-"
    try:
        if abs(float(x)) >= 1000:
            return f"{float(x):,.0f}"
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return str(x)

# ==========================================================
# DISPLAY TABLE
# ==========================================================
def make_display_table(master_df):

    disp = master_df.copy()
    # Content weight
    disp["Content Weight"] = (
        disp["Content Weight"]
        .apply(
            lambda x:
                f"{int(x):,}"
                if pd.notna(x)
                else "-"
        )
    )


    # Losses
    for column in [

        "Final Content Loss",
        "Final Style Loss",
        "Final Total Loss",
        "Best Total Loss",

    ]:

        disp[column] = (
            disp[column]
            .apply(fmt_num)
        )


    # Adaptive weights
    for column in [

        "Final Content Weight",
        "Final Style Weight",

    ]:

        disp[column] = (
            disp[column]
            .apply(fmt_num)
        )


    # Iterations
    disp["Iterations Logged"] = (
        disp["Iterations Logged"]
        .apply(
            lambda x:
                f"{int(x)}"
                if pd.notna(x)
                else "-"
        )
    )


    disp["Best Iteration"] = (
        disp["Best Iteration"]
        .apply(
            lambda x:
                f"{int(x)}"
                if pd.notna(x)
                else "-"
        )
    )


    # Remove source from visual
    disp = disp.drop(
        columns=["Source"]
    )

    return disp

# ==========================================================
# RENDER PNG
# ==========================================================

def render_png(
    disp_df,
    out_path
):

    n_rows, n_cols = (
        disp_df.shape
    )

    fig_w = 30
    fig_h = max(
        5,
        0.65 * (n_rows + 2)
    )

    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h)
    )

    ax.axis("off")

    # ======================================================
    # GROUP COLOURS
    # ======================================================

    group_colors = {

        "Optimizer x Model":
            "#e8f0fe",

        "Adaptive Weighting":
            "#fef7e0",

        "Content-Weight Sweep":
            "#e6f4ea",
    }

    # ======================================================
    # CREATE TABLE
    # ======================================================

    table = ax.table(

        cellText=disp_df.values,
        colLabels=disp_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )


    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        8.5
    )

    table.scale(
        1,
        1.8
    )

    # ======================================================
    # AUTOMATIC COLUMN WIDTH
    # ======================================================

    table.auto_set_column_width(
        col=list(range(n_cols))
    )

    # ======================================================
    # HEADER
    # ======================================================

    for j in range(n_cols):

        cell = table[
            0,
            j
        ]

        cell.set_facecolor(
            "#333333"
        )

        cell.set_text_props(
            color="white",
            weight="bold"
        )

    # ======================================================
    # DATA ROWS
    # ======================================================

    for i in range(n_rows):

        group = (
            disp_df.iloc[i]["Group"]
        )


        color = group_colors.get(
            group,
            "#ffffff"
        )


        for j in range(n_cols):

            cell = table[
                i + 1,
                j
            ]


            cell.set_facecolor(
                color
            )


            if (
                disp_df.columns[j]
                == "Experiment"
            ):

                cell.set_text_props(
                    weight="bold"
                )

    # ======================================================
    # TITLE
    # ======================================================

    fig.suptitle(

        "Neural Style Transfer — "
        "Master Experiment Comparison",

        fontsize=17,

        weight="bold",

        y=0.97
    )

    plt.subplots_adjust(
        top=0.88,
        bottom=0.05,
        left=0.01,
        right=0.99
    )

    fig.savefig(

        out_path,

        dpi=200,

        bbox_inches="tight"
    )

    plt.close(fig)

# ==========================================================
# RENDER HTML
# ==========================================================

def render_html(
    master_df,
    out_path
):

    numeric_loss_cols = [

        "Final Content Loss",
        "Final Style Loss",
        "Final Total Loss",
        "Best Total Loss",
    ]


    styled = (

        master_df.style

        .background_gradient(

            subset=numeric_loss_cols,

            cmap="RdYlGn_r"
        )

        .format(

            precision=2,

            na_rep="-"
        )

        .set_caption(

            "Neural Style Transfer — "
            "Master Experiment Comparison"
        )

        .set_table_styles([

            {
                "selector": "caption",

                "props": [

                    ("font-size", "18px"),

                    ("font-weight", "bold"),

                    ("padding", "10px"),
                ],
            },

            {
                "selector": "th",

                "props": [

                    ("background-color", "#333"),

                    ("color", "white"),

                    ("padding", "8px"),

                    ("white-space", "nowrap"),
                ],
            },

            {
                "selector": "td",

                "props": [

                    ("padding", "8px"),

                    ("white-space", "nowrap"),
                ],
            },
        ])
    )


    styled.to_html(
        out_path
    )

# ==========================================================
# MAIN
# ==========================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    print()
    print("=" * 70)
    print(
        "NEURAL STYLE TRANSFER — "
        "MASTER EXPERIMENT COMPARISON"
    )
    print("=" * 70)
    print()


    master_df = build_master_table(
        EXPERIMENTS
    )


    # ======================================================
    # OUTPUT PATHS
    # ======================================================

    csv_out = os.path.join(
        OUTPUT_DIR,
        "master_comparison_table.csv"
    )


    html_out = os.path.join(
        OUTPUT_DIR,
        "master_comparison_table.html"
    )


    png_out = os.path.join(
        OUTPUT_DIR,
        "master_comparison_table.png"
    )


    # ======================================================
    # SAVE CSV
    # ======================================================

    master_df.to_csv(
        csv_out,
        index=False
    )


    print()
    print(
        f"[SAVED] {csv_out}"
    )


    # ======================================================
    # SAVE HTML
    # ======================================================

    render_html(
        master_df,
        html_out
    )


    print(
        f"[SAVED] {html_out}"
    )


    # ======================================================
    # SAVE PNG
    # ======================================================

    display_df = (
        make_display_table(
            master_df
        )
    )


    render_png(
        display_df,
        png_out
    )


    print(
        f"[SAVED] {png_out}"
    )


    # ======================================================
    # FINAL CHECK
    # ======================================================

    print()
    print("=" * 70)
    print("FINAL EXPERIMENT STATUS")
    print("=" * 70)


    for _, row in master_df.iterrows():

        status = (
            "OK"
            if pd.notna(
                row["Final Total Loss"]
            )
            else "MISSING"
        )


        print(
            f"[{status:7s}] "
            f"{row['Experiment']}"
        )


    print()
    print("Done.")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()