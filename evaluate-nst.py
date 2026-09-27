import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.definitions.vgg_nets import Vgg16
from utils.utils import gram_matrix

# =========================
# SETTINGS
# =========================

CONTENT_IMAGE = "data/content-images/mount-fuji.jpg"
STYLE_IMAGE = "data/style-images/ancient-chinese-style.jpg"
EXPERIMENT_DIR = "experiments (content_weight_vgg16_LBFGS)"
ADAPTIVE_DIR = "experiments (adaptive_weighting)"
PLOT_OUTPUT_DIR = "experiments (content_weight_vgg16_LBFGS)/_plots"

DEVICE = torch.device("cpu")

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255)),
    transforms.Normalize(
        mean=IMAGENET_MEAN_255,
        std=IMAGENET_STD_NEUTRAL
    )
])

# =========================
# IMAGE LOADING
# =========================

def load_image(path, target_size=None):
    image = Image.open(path).convert("RGB")

    if target_size is not None:
        image = image.resize(target_size)

    image = transform(image)

    return image.unsqueeze(0).to(DEVICE)

# =========================
# LOCATE ADAPTIVE-WEIGHTING RUNS (VGG16 + VGG19)
# =========================

def _find_jpgs_recursive(root):
    found = []

    if not os.path.exists(root):
        return found

    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".jpg"):
                full_path = os.path.join(dirpath, f)
                found.append((full_path, os.path.getmtime(full_path)))

    return found


def find_adaptive_image(model_filter):
    all_jpgs = _find_jpgs_recursive(ADAPTIVE_DIR)

    matches = [
        (path, mtime) for path, mtime in all_jpgs
        if model_filter.lower() in path.lower()
    ]

    if not matches:
        return None, None

    # most recently modified match wins
    matches.sort(key=lambda item: item[1], reverse=True)
    best_path = matches[0][0]

    # label = the immediate parent folder name, for a readable legend
    label = os.path.basename(os.path.dirname(best_path))

    return label, best_path


ADAPTIVE_VGG16_LABEL, ADAPTIVE_VGG16_IMAGE = find_adaptive_image("vgg16")
ADAPTIVE_VGG19_LABEL, ADAPTIVE_VGG19_IMAGE = find_adaptive_image("vgg19")

# =========================
# LOAD VGG16
# =========================

print("Loading VGG16...")

vgg = Vgg16(
    requires_grad=False,
    show_progress=True
).to(DEVICE).eval()

# =========================
# LOAD CONTENT + STYLE
# =========================

content = load_image(CONTENT_IMAGE)
style = load_image(STYLE_IMAGE)

# =========================
# FIND GENERATED IMAGES
# =========================

runs = [
    "00-baseline_100000",
    "01_cw_50000",
    "02_cw_75000",
    "03_cw_150000",
    "04_cw_200000"
]

# =========================
# EVALUATION FUNCTION
# =========================

def evaluate_image(generated_path):

    generated_pil = Image.open(
        generated_path
    ).convert("RGB")

    generated_size = generated_pil.size

    # Resize reference images to generated image size
    content = load_image(
        CONTENT_IMAGE,
        generated_size
    )

    style = load_image(
        STYLE_IMAGE,
        generated_size
    )

    generated = load_image(
        generated_path,
        generated_size
    )

    # Extract VGG16 features
    with torch.no_grad():

        content_features = vgg(content)
        style_features = vgg(style)
        generated_features = vgg(generated)

    # =========================
    # CONTENT DISTANCE
    # =========================

    content_target = content_features[
        vgg.content_feature_maps_index
    ]

    content_generated = generated_features[
        vgg.content_feature_maps_index
    ]

    content_distance = torch.mean(
        (content_generated - content_target) ** 2
    ).item()

    # =========================
    # STYLE DISTANCE
    # =========================

    style_distance = 0.0

    for index in vgg.style_feature_maps_indices:

        style_target = gram_matrix(
            style_features[index]
        )

        style_generated = gram_matrix(
            generated_features[index]
        )

        style_distance += torch.mean(
            (style_generated - style_target) ** 2
        ).item()

    style_distance /= len(
        vgg.style_feature_maps_indices
    )

    return content_distance, style_distance

# =========================
# RUN ALL EXPERIMENTS (fixed content-weight sweep)
# =========================

print("\n==========================================")
print("CONTENT-WEIGHT QUANTITATIVE EVALUATION")
print("==========================================")

results = []


for run in runs:

    run_path = os.path.join(
        EXPERIMENT_DIR,
        run
    )

    if not os.path.exists(run_path):
        print(f"\n[SKIPPED] {run} folder not found.")
        continue

    # Find generated JPG
    jpg_files = [
        f for f in os.listdir(run_path)
        if f.lower().endswith(".jpg")
    ]

    if not jpg_files:
        print(f"\n[SKIPPED] No JPG found in {run}")
        continue

    generated_path = os.path.join(
        run_path,
        jpg_files[0]
    )

    print(f"\nEvaluating: {run}")

    content_distance, style_distance = evaluate_image(
        generated_path
    )

    # Extract content weight from folder name
    if "baseline" in run:
        content_weight = 100000
    else:
        content_weight = int(
            run.split("_")[-1]
        )

    results.append(
        (
            content_weight,
            content_distance,
            style_distance
        )
    )

    print(
        f"Content distance: {content_distance:.6f}"
    )

    print(
        f"Style distance:   {style_distance:.6f}"
    )

# =========================
# EVALUATE ADAPTIVE RUNS (VGG16 + VGG19)
# =========================

adaptive_results = []  # list of (label, content_distance, style_distance)

for tag, label, image_path in [
    ("adaptive_vgg16", ADAPTIVE_VGG16_LABEL, ADAPTIVE_VGG16_IMAGE),
    ("adaptive_vgg19", ADAPTIVE_VGG19_LABEL, ADAPTIVE_VGG19_IMAGE),
]:

    if image_path and os.path.exists(image_path):

        print(f"\nEvaluating: {tag} ({label})")

        content_distance, style_distance = evaluate_image(image_path)

        adaptive_results.append((tag, content_distance, style_distance))

        results.append((tag, content_distance, style_distance))

        print(f"Content distance: {content_distance:.6f}")
        print(f"Style distance:   {style_distance:.6f}")

    else:
        print(f"\n[SKIPPED] No adaptive image found for {tag} under: {ADAPTIVE_DIR}")

# =========================
# FINAL TABLE
# =========================

print("\n\n==========================================")
print("FINAL RESULTS")
print("==========================================")

LABEL_WIDTH = 30
DIST_WIDTH = 20

print(
    f"{'Content Weight':<{LABEL_WIDTH}}"
    f"{'Content Distance':<{DIST_WIDTH}}"
    f"{'Style Distance':<{DIST_WIDTH}}"
)

print("-" * (LABEL_WIDTH + DIST_WIDTH * 2))

for content_weight, content_distance, style_distance in results:

    print(
        f"{str(content_weight):<{LABEL_WIDTH}}"
        f"{content_distance:<{DIST_WIDTH}.6f}"
        f"{style_distance:<{DIST_WIDTH}.6f}"
    )

print("\nLower distance = better similarity.")
print("==========================================")

# =========================
# LINE GRAPH VISUALIZATIONS
# =========================

def plot_line_comparisons(results, adaptive_results, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    numeric_results = [
        r for r in results
        if isinstance(r[0], int)
    ]
    numeric_results.sort(key=lambda r: r[0])

    if not numeric_results:
        print("[SKIPPED] No numeric content-weight results to plot.")
        return

    weights = [r[0] for r in numeric_results]
    content_distances = [r[1] for r in numeric_results]
    style_distances = [r[2] for r in numeric_results]

    adaptive_styles = {
        "adaptive_vgg16": {"color": "tab:green", "linestyle": "--"},
        "adaptive_vgg19": {"color": "tab:purple", "linestyle": "-."},
    }

    # -------------------------------------------------
    # Plot 1: both metrics on one graph, dual y-axis
    # -------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color1 = "tab:blue"
    ax1.set_xlabel("Content Weight")
    ax1.set_ylabel("Content Distance", color=color1)
    line1 = ax1.plot(
        weights, content_distances,
        marker="o", color=color1, label="Content Distance"
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Style Distance", color=color2)
    line2 = ax2.plot(
        weights, style_distances,
        marker="s", color=color2, label="Style Distance"
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    for tag, content_val, style_val in adaptive_results:
        style_kwargs = adaptive_styles.get(tag, {"color": "gray", "linestyle": "--"})
        ax1.axhline(
            content_val, alpha=0.6, label=f"{tag} content",
            **style_kwargs
        )
        ax2.axhline(
            style_val, alpha=0.6, label=f"{tag} style",
            **style_kwargs
        )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", fontsize=8)

    ax1.set_title("Content vs Style Distance across Content Weight (dual axis)")
    fig.tight_layout()

    dual_axis_path = os.path.join(output_dir, "content_weight_dual_axis.png")
    fig.savefig(dual_axis_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {dual_axis_path}")

    # -------------------------------------------------
    # Plot 2: normalized (0-1) both metrics, single axis
    # so the crossover / trade-off point is visible directly
    # -------------------------------------------------

    def normalize(values):
        lo, hi = min(values), max(values)
        if hi == lo:
            return [0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    norm_content = normalize(content_distances)
    norm_style = normalize(style_distances)

    fig2, ax = plt.subplots(figsize=(9, 6))

    ax.plot(weights, norm_content, marker="o", label="Content Distance (normalized)")
    ax.plot(weights, norm_style, marker="s", label="Style Distance (normalized)")

    ax.set_xlabel("Content Weight")
    ax.set_ylabel("Normalized Distance (0 = best, 1 = worst)")
    ax.set_title("Normalized Content vs Style Distance Trade-off")
    ax.legend(fontsize=8)
    fig2.tight_layout()

    normalized_path = os.path.join(output_dir, "content_weight_normalized.png")
    fig2.savefig(normalized_path, dpi=150)
    plt.close(fig2)
    print(f"[SAVED] {normalized_path}")

    # -------------------------------------------------
    # Plot 3: side-by-side subplots, one line each,
    # easiest to read individually for a report
    # -------------------------------------------------

    fig3, (axc, axs) = plt.subplots(1, 2, figsize=(12, 5))

    axc.plot(weights, content_distances, marker="o", color="tab:blue")
    axc.set_xlabel("Content Weight")
    axc.set_ylabel("Content Distance")
    axc.set_title("Content Distance vs Content Weight")

    axs.plot(weights, style_distances, marker="s", color="tab:red")
    axs.set_xlabel("Content Weight")
    axs.set_ylabel("Style Distance")
    axs.set_title("Style Distance vs Content Weight")

    for tag, content_val, style_val in adaptive_results:
        style_kwargs = adaptive_styles.get(tag, {"color": "gray", "linestyle": "--"})
        axc.axhline(content_val, alpha=0.6, label=tag, **style_kwargs)
        axs.axhline(style_val, alpha=0.6, label=tag, **style_kwargs)

    if adaptive_results:
        axc.legend(fontsize=7)
        axs.legend(fontsize=7)

    fig3.tight_layout()

    side_by_side_path = os.path.join(output_dir, "content_weight_side_by_side.png")
    fig3.savefig(side_by_side_path, dpi=150)
    plt.close(fig3)
    print(f"[SAVED] {side_by_side_path}")

    # -------------------------------------------------
    # Plot 4: direct adaptive vgg16 vs vgg19 grouped bar chart
    # -------------------------------------------------

    if len(adaptive_results) >= 2:

        fig4, ax4 = plt.subplots(figsize=(7, 5))

        tags = [r[0] for r in adaptive_results]
        adaptive_content_vals = [r[1] for r in adaptive_results]
        adaptive_style_vals = [r[2] for r in adaptive_results]

        x = range(len(tags))
        width = 0.35

        ax4.bar([i - width / 2 for i in x], adaptive_content_vals,
                width, label="Content Distance", color="tab:blue")
        ax4.bar([i + width / 2 for i in x], adaptive_style_vals,
                width, label="Style Distance", color="tab:red")

        ax4.set_xticks(list(x))
        ax4.set_xticklabels(tags)
        ax4.set_ylabel("Distance")
        ax4.set_title("Adaptive Weighting: VGG16 vs VGG19")
        ax4.legend()
        fig4.tight_layout()

        adaptive_compare_path = os.path.join(output_dir, "adaptive_vgg16_vs_vgg19.png")
        fig4.savefig(adaptive_compare_path, dpi=150)
        plt.close(fig4)
        print(f"[SAVED] {adaptive_compare_path}")

    elif adaptive_results:
        print("[SKIPPED] Only one adaptive result found — need both "
              "vgg16 and vgg19 adaptive runs to plot the comparison bar chart.")


print("\n==========================================")
print("GENERATING LINE GRAPHS")
print("==========================================")

plot_line_comparisons(results, adaptive_results, PLOT_OUTPUT_DIR)

print(f"\nAll plots saved to: {PLOT_OUTPUT_DIR}")