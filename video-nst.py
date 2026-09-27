import os
import glob
import subprocess
import cv2
import numpy as np
import torch
import torch_directml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = r"C:\Users\Wei Keong\Desktop\Y3S2\final-year-project\pytorch-neural-style-transfer-master"

CONTENT_VIDEO_DIR = os.path.join(
    BASE_DIR, "data", "content-videos"
)

STYLE_IMAGE_DIR = os.path.join(
    BASE_DIR, "data", "style-images"
)

OUTPUT_VIDEO_DIR = os.path.join(
    BASE_DIR, "data", "output-videos"
)


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

VGG_MODEL = "vgg16"

CONTENT_LAYER = "relu2_1"

STYLE_LAYERS = [
    "relu1_1",
    "relu2_1",
    "relu3_1",
    "relu4_1",
    "relu5_1"
]


# -----------------------------------------------------------------------------
# VIDEO SETTINGS
# -----------------------------------------------------------------------------

FIRST_FRAME_ITERATIONS = 150
LATER_FRAME_ITERATIONS = 75
FRAME_WIDTH = 256
STYLE_IMAGE_SIZE = 384
USE_PREVIOUS_FRAME = True
GHOST_FRAME_BLEND = 0.15
DEBUG_FIRST_FRAME_ONLY = False
MAX_FRAMES = None
SAVE_PROGRESS_SNAPSHOTS = True
SNAPSHOT_INTERVAL = 10
PROGRESS_VIDEO_FPS = 5

# -----------------------------------------------------------------------------
# LOSS WEIGHTS
# -----------------------------------------------------------------------------

CONTENT_WEIGHT = 5e5
STYLE_WEIGHT = 1e7
TV_WEIGHT = 0.1

# -----------------------------------------------------------------------------
# LBFGS SETTINGS
# -----------------------------------------------------------------------------

LBFGS_HISTORY_SIZE = 5
LBFGS_LEARNING_RATE = 0.5
LBFGS_TOLERANCE_GRAD = 1e-7
LBFGS_TOLERANCE_CHANGE = 1e-9

# =============================================================================
# DEVICE
# =============================================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    DEVICE_NAME = "CUDA"

else:

    DEVICE = torch_directml.device()

    DEVICE_NAME = "DirectML"


# =============================================================================
# VGG NORMALISATION
# =============================================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
    device=DEVICE
).view(1, 3, 1, 1)


STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
    device=DEVICE
).view(1, 3, 1, 1)


# =============================================================================
# DEVICE INFORMATION
# =============================================================================

def print_device():

    print("=" * 70)
    print("DEVICE CHECK")
    print("=" * 70)

    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(f"Device:         {DEVICE}")
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
        print(f"CUDA:           {torch.version.cuda}")

    else:

        print(f"DirectML:       {DEVICE}")
        print("[INFO] CUDA is unavailable.")
        print("[INFO] Using DirectML for AMD GPU acceleration.")

    print("=" * 70)


# =============================================================================
# VGG16 FEATURE EXTRACTOR
# =============================================================================

class VGG16Features(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT
        ).features

        for parameter in self.features.parameters():

            parameter.requires_grad = False

    def forward(self, x):

        outputs = {}

        for i, layer in enumerate(self.features):

            x = layer(x)

            if i == 1:
                outputs["relu1_1"] = x
            elif i == 6:
                outputs["relu2_1"] = x
            elif i == 11:
                outputs["relu3_1"] = x
            elif i == 18:
                outputs["relu4_1"] = x
            elif i == 25:
                outputs["relu5_1"] = x
        return outputs


# =============================================================================
# NORMALISATION
# =============================================================================

def normalize(x):
    return (x - MEAN) / STD

# =============================================================================
# SAFE IMAGE CLAMP
# =============================================================================

def safe_image(x):
    return torch.clamp(x, 0.0, 1.0)

# =============================================================================
# GRAM MATRIX
# =============================================================================

def gram_matrix(x):
    b, c, h, w = x.shape

    features = x.reshape(
        b,
        c,
        h * w
    )

    gram = torch.bmm(
        features,
        features.transpose(1, 2)
    )

    gram = gram / float(c * h * w)

    return gram


# =============================================================================
# STABLE TOTAL VARIATION
# =============================================================================

def total_variation(x):
    eps = 1e-6

    horizontal = (
        x[:, :, :, 1:]
        -
        x[:, :, :, :-1]
    )

    vertical = (
        x[:, :, 1:, :]
        -
        x[:, :, :-1, :]
    )

    horizontal_loss = torch.sqrt(
        horizontal * horizontal + eps
    ).mean()

    vertical_loss = torch.sqrt(
        vertical * vertical + eps
    ).mean()

    return horizontal_loss + vertical_loss


# =============================================================================
# FINITE CHECK
# =============================================================================

def tensor_is_finite(x):

    try:

        value = torch.isfinite(x).all()

        return bool(
            value.detach().cpu().item()
        )

    except Exception:

        return True


# =============================================================================
# FRAME -> TENSOR
# =============================================================================

def frame_to_tensor(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    array = np.ascontiguousarray(
        rgb
    )

    tensor = torch.from_numpy(
        array
    ).float()

    tensor = tensor / 255.0

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(
        DEVICE
    )


# =============================================================================
# TENSOR -> FRAME
# =============================================================================

def tensor_to_frame(x):

    with torch.no_grad():

        x = safe_image(x)

        x = x.squeeze(0)

        x = x.permute(
            1,
            2,
            0
        )

        x = x * 255.0

        x = x.byte()

        x = x.cpu().numpy()

    return cv2.cvtColor(
        x,
        cv2.COLOR_RGB2BGR
    )


# =============================================================================
# RESIZE VIDEO FRAME
# =============================================================================

def resize_frame(frame):

    height, width = frame.shape[:2]

    scale = FRAME_WIDTH / float(width)

    new_width = FRAME_WIDTH

    new_height = int(
        height * scale
    )

    if new_width % 2 != 0:
        new_width -= 1

    if new_height % 2 != 0:
        new_height -= 1

    if width == new_width and height == new_height:

        return frame

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# =============================================================================
# LOAD STYLE IMAGE
# =============================================================================

def load_style(path):

    image = Image.open(
        path
    ).convert("RGB")

    image.thumbnail(
        (
            STYLE_IMAGE_SIZE,
            STYLE_IMAGE_SIZE
        ),
        Image.Resampling.LANCZOS
    )

    image = image.resize(
        (
            STYLE_IMAGE_SIZE,
            STYLE_IMAGE_SIZE
        ),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array /= 255.0

    tensor = torch.from_numpy(
        array
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(
        DEVICE
    )


# =============================================================================
# PRECOMPUTE STYLE TARGETS
# =============================================================================

def style_targets(model, style_tensor):

    with torch.no_grad():

        features = model(
            normalize(
                style_tensor
            )
        )

        targets = {}

        for layer in STYLE_LAYERS:

            targets[layer] = gram_matrix(
                features[layer]
            ).detach()

    return targets


# =============================================================================
# COMPUTE LOSSES
# =============================================================================

def compute_losses(
    generated,
    content_target,
    style_target_dict,
    model
):

    image = safe_image(
        generated
    )

    features = model(
        normalize(image)
    )

    # -------------------------------------------------------------------------
    # CONTENT LOSS
    # -------------------------------------------------------------------------

    content_loss = F.mse_loss(
        features[CONTENT_LAYER],
        content_target,
        reduction="mean"
    )

    # -------------------------------------------------------------------------
    # STYLE LOSS
    # -------------------------------------------------------------------------

    style_loss = torch.zeros(
        (),
        device=DEVICE,
        dtype=torch.float32
    )

    for layer in STYLE_LAYERS:

        current_gram = gram_matrix(
            features[layer]
        )

        target_gram = style_target_dict[
            layer
        ]

        layer_loss = F.mse_loss(
            target_gram[0],
            current_gram[0],
            reduction="sum"
        )

        style_loss = (
            style_loss
            +
            layer_loss
        )

    style_loss = (
        style_loss
        /
        len(STYLE_LAYERS)
    )

    # -------------------------------------------------------------------------
    # TV LOSS
    # -------------------------------------------------------------------------

    tv_loss = total_variation(
        image
    )

    # -------------------------------------------------------------------------
    # TOTAL LOSS
    # -------------------------------------------------------------------------

    total_loss = (
        CONTENT_WEIGHT * content_loss
        +
        STYLE_WEIGHT * style_loss
        +
        TV_WEIGHT * tv_loss
    )

    return (
        total_loss,
        content_loss,
        style_loss,
        tv_loss
    )


# =============================================================================
# STYLIZE ONE FRAME
# =============================================================================

def stylize_frame(
    content,
    model,
    style_targets_dict,
    previous,
    iterations,
    frame_number,
    snapshot_dir=None
):

    # =========================================================================
    # CONTENT TARGET
    # =========================================================================

    with torch.no_grad():

        content_features = model(
            normalize(
                content
            )
        )

        content_target = content_features[
            CONTENT_LAYER
        ].detach()

    # =========================================================================
    # INITIAL IMAGE
    # =========================================================================

    if (
        USE_PREVIOUS_FRAME
        and
        previous is not None
    ):
        generated = (
            (1.0 - GHOST_FRAME_BLEND) * content
            +
            GHOST_FRAME_BLEND * previous
        ).clone().detach()

        initialization_name = (
            f"content frame + {GHOST_FRAME_BLEND:.2f} "
            f"previous-frame blend"
        )

    else:

        generated = content.clone().detach()

        initialization_name = (
            "content frame"
        )

    generated = safe_image(
        generated
    ).detach()

    generated.requires_grad_(
        True
    )

    # =========================================================================
    # LBFGS
    # =========================================================================

    optimizer = optim.LBFGS(
        [generated],

        lr=LBFGS_LEARNING_RATE,

        max_iter=1,

        history_size=LBFGS_HISTORY_SIZE,

        tolerance_grad=LBFGS_TOLERANCE_GRAD,

        tolerance_change=LBFGS_TOLERANCE_CHANGE,

        line_search_fn="strong_wolfe"
    )

    last_good_image = (
        generated.detach().clone()
    )

    last_losses = None

    clip_count = 0

    clip_max_norm = 0.0

    for iteration in range(iterations):

        def closure():

            nonlocal clip_count, clip_max_norm

            optimizer.zero_grad()

            (
                total_loss,
                content_loss,
                style_loss,
                tv_loss
            ) = compute_losses(
                generated,
                content_target,
                style_targets_dict,
                model
            )

            # -----------------------------------------------------------------
            # Numerical safety
            # -----------------------------------------------------------------

            if not tensor_is_finite(
                total_loss
            ):

                print(
                    f"\n[WARNING] Non-finite loss detected at "
                    f"frame {frame_number}, "
                    f"iteration {iteration + 1}."
                )

                safe_loss = (
                    content_loss.new_tensor(
                        0.0
                    )
                )

                return safe_loss

            total_loss.backward()

            if generated.grad is not None:

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [generated],
                    max_norm=1e6
                )

                if grad_norm > 1e6:

                    clip_count += 1

                    clip_max_norm = max(
                        clip_max_norm,
                        float(grad_norm)
                    )

            return total_loss

        # ---------------------------------------------------------------------
        # Perform one LBFGS update.
        # ---------------------------------------------------------------------

        old_image = (
            generated.detach().clone()
        )

        optimizer.step(
            closure
        )

        # ---------------------------------------------------------------------
        # Check resulting image.
        # ---------------------------------------------------------------------

        with torch.no_grad():

            finite = tensor_is_finite(
                generated
            )

            if not finite:

                print(
                    f"\n[WARNING] Non-finite image detected at "
                    f"frame {frame_number}, "
                    f"iteration {iteration + 1}."
                )

                generated.copy_(
                    old_image
                )

                optimizer = optim.LBFGS(
                    [generated],

                    lr=(
                        LBFGS_LEARNING_RATE
                        *
                        0.5
                    ),

                    max_iter=1,

                    history_size=LBFGS_HISTORY_SIZE,

                    tolerance_grad=(
                        LBFGS_TOLERANCE_GRAD
                    ),

                    tolerance_change=(
                        LBFGS_TOLERANCE_CHANGE
                    ),

                    line_search_fn="strong_wolfe"
                )

            else:

                generated.clamp_(
                    0.0,
                    1.0
                )

                last_good_image = (
                    generated.detach().clone()
                )

        # ---------------------------------------------------------------------
        # Save a frame-0 optimisation-progress snapshot.
        # ---------------------------------------------------------------------

        if (
            snapshot_dir is not None
            and
            (
                iteration == 0
                or
                (iteration + 1) % SNAPSHOT_INTERVAL == 0
                or
                iteration == iterations - 1
            )
        ):

            os.makedirs(
                snapshot_dir,
                exist_ok=True
            )

            snapshot_frame = tensor_to_frame(
                last_good_image
            )

            snapshot_path = os.path.join(
                snapshot_dir,
                f"{iteration + 1:04d}.png"
            )

            cv2.imwrite(
                snapshot_path,
                snapshot_frame
            )

        # ---------------------------------------------------------------------
        # Print progress for first frame.
        # ---------------------------------------------------------------------

        if frame_number == 0:

            if (
                iteration == 0
                or
                (iteration + 1) % 25 == 0
                or
                iteration == iterations - 1
            ):

                with torch.no_grad():

                    (
                        total_loss,
                        content_loss,
                        style_loss,
                        tv_loss
                    ) = compute_losses(
                        generated,
                        content_target,
                        style_targets_dict,
                        model
                    )

                    total_value = (
                        total_loss.item()
                    )

                    content_value = (
                        content_loss.item()
                    )

                    style_value = (
                        style_loss.item()
                    )

                    tv_value = (
                        tv_loss.item()
                    )

                    weighted_content = (
                        CONTENT_WEIGHT
                        *
                        content_value
                    )

                    weighted_style = (
                        STYLE_WEIGHT
                        *
                        style_value
                    )

                    weighted_tv = (
                        TV_WEIGHT
                        *
                        tv_value
                    )

                    print(
                        f"[FRAME 0] "
                        f"iter {iteration + 1:03d}/{iterations} | "
                        f"total={total_value:12.2f} | "
                        f"content={weighted_content:10.2f} | "
                        f"style={weighted_style:10.2f} | "
                        f"tv={weighted_tv:8.4f}"
                    )

                    last_losses = (
                        total_value,
                        content_value,
                        style_value,
                        tv_value
                    )

    # =========================================================================
    # FINAL LOSS REPORT
    # =========================================================================

    with torch.no_grad():

        (
            total_loss,
            content_loss,
            style_loss,
            tv_loss
        ) = compute_losses(
            generated,
            content_target,
            style_targets_dict,
            model
        )

        total_value = (
            total_loss.item()
        )

        content_value = (
            content_loss.item()
        )

        style_value = (
            style_loss.item()
        )

        tv_value = (
            tv_loss.item()
        )

        weighted_content = (
            CONTENT_WEIGHT
            *
            content_value
        )

        weighted_style = (
            STYLE_WEIGHT
            *
            style_value
        )

        weighted_tv = (
            TV_WEIGHT
            *
            tv_value
        )

        contribution_sum = (
            weighted_content
            +
            weighted_style
            +
            weighted_tv
        )

        if contribution_sum > 0:

            style_percentage = (
                weighted_style
                /
                contribution_sum
                *
                100.0
            )

        else:

            style_percentage = 0.0

    if frame_number == 0:

        print()
        print("-" * 70)
        print("FIRST FRAME RESULT")
        print("-" * 70)

        print(
            f"Initialization:      "
            f"{initialization_name}"
        )

        print(
            f"Iterations:          "
            f"{iterations}"
        )

        print(
            f"Raw content loss:    "
            f"{content_value:.8f}"
        )

        print(
            f"Raw style loss:      "
            f"{style_value:.8f}"
        )

        print(
            f"Weighted content:    "
            f"{weighted_content:.4f}"
        )

        print(
            f"Weighted style:      "
            f"{weighted_style:.4f}"
        )

        print(
            f"Weighted TV:         "
            f"{weighted_tv:.4f}"
        )

        print(
            f"TOTAL LOSS:          "
            f"{total_value:.4f}"
        )

        print(
            f"Style contribution:  "
            f"{style_percentage:.2f}%"
        )

        print("-" * 70)

    if clip_count > 0:

        print(
            f"[FRAME {frame_number}] Gradient clipping engaged "
            f"{clip_count} time(s), max norm seen: "
            f"{clip_max_norm:,.0f}"
        )

    return last_good_image.detach()


# =============================================================================
# DEBUG IMAGE SAVING
# =============================================================================

def save_debug(
    content,
    output,
    video_name,
    style_name
):

    debug_dir = os.path.join(
        OUTPUT_VIDEO_DIR,
        "_debug_frame0"
    )

    os.makedirs(
        debug_dir,
        exist_ok=True
    )

    content_path = os.path.join(
        debug_dir,
        f"{video_name}_style_{style_name}_content.png"
    )

    stylized_path = os.path.join(
        debug_dir,
        f"{video_name}_style_{style_name}_stylized.png"
    )

    cv2.imwrite(
        content_path,
        content
    )

    cv2.imwrite(
        stylized_path,
        output
    )

    print()
    print("[DEBUG] Content:")
    print(
        f"        {content_path}"
    )

    print("[DEBUG] Stylized:")
    print(
        f"        {stylized_path}"
    )


# =============================================================================
# BUILD FRAME-0 OPTIMISATION PROGRESS VIDEO
# =============================================================================

# =============================================================================
# TRANSCODE TO H.264 (VS CODE / BROWSER COMPATIBLE)
# =============================================================================

def _transcode_to_h264(raw_path, final_path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", raw_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            final_path
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0 or not os.path.exists(final_path):

        print(
            f"[WARNING] FFmpeg transcode to H.264 failed "
            f"(is ffmpeg on PATH?). Keeping raw mp4v file instead: "
            f"{raw_path}"
        )

        print(
            f"          FFmpeg stderr: {result.stderr[-500:]}"
        )

        if raw_path != final_path and os.path.exists(raw_path):

            os.replace(
                raw_path,
                final_path
            )

        return

    if os.path.exists(raw_path):

        os.remove(
            raw_path
        )


# =============================================================================
# VERIFY A WRITTEN VIDEO IS ACTUALLY PLAYABLE
# =============================================================================

def _report_video_write_result(
    output_path,
    label,
    extra=""
):

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:

        print(
            f"[ERROR] {label} was not created (0 bytes): {output_path}"
        )

        return

    check = cv2.VideoCapture(output_path)

    readable, _ = check.read()

    frame_count = int(
        check.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    check.release()

    if not readable or frame_count <= 0:

        print(
            f"[ERROR] {label} was written but contains no readable "
            f"frames (likely an encoder/dimension mismatch — e.g. an "
            f"odd width or height): {output_path}"
        )

        return

    suffix = f" ({extra})" if extra else ""

    print(
        f"[SAVED] {label}{suffix} -> {output_path}"
    )


# =============================================================================
# BUILD FRAME-0 OPTIMISATION PROGRESS VIDEO
# =============================================================================

def build_progress_video(
    snapshot_dir,
    output_path,
    fps=PROGRESS_VIDEO_FPS
):

    files = sorted(
        glob.glob(
            os.path.join(
                snapshot_dir,
                "*.png"
            )
        )
    )

    if not files:

        print(
            f"[WARNING] No frame-0 snapshots found in {snapshot_dir}, "
            f"skipping progress video."
        )

        return

    first = cv2.imread(
        files[0]
    )

    height, width = first.shape[:2]

    raw_path = output_path + ".raw_mp4v.mp4"

    writer = cv2.VideoWriter(
        raw_path,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (width, height)
    )

    if not writer.isOpened():

        print(
            f"[ERROR] Could not create progress video: {output_path}"
        )

        return

    for path in files:

        writer.write(
            cv2.imread(path)
        )

    # Hold on the final frame for ~2 seconds.
    last_frame = cv2.imread(
        files[-1]
    )

    for _ in range(fps * 2):

        writer.write(
            last_frame
        )

    writer.release()

    _transcode_to_h264(
        raw_path,
        output_path
    )

    _report_video_write_result(
        output_path,
        "Frame-0 optimisation progress video",
        extra=f"{len(files)} snapshots"
    )


# =============================================================================
# BUILD FINAL VIDEO FROM CHECKPOINTED FRAMES
# =============================================================================

def build_final_video(
    frames_dir,
    output_path,
    fps
):

    files = sorted(
        glob.glob(
            os.path.join(
                frames_dir,
                "frame_*.png"
            )
        )
    )

    if not files:

        print(
            f"[WARNING] No checkpointed frames found in {frames_dir}, "
            f"cannot build output video."
        )

        return

    first = cv2.imread(
        files[0]
    )

    height, width = first.shape[:2]

    raw_path = output_path + ".raw_mp4v.mp4"

    writer = cv2.VideoWriter(
        raw_path,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (width, height)
    )

    if not writer.isOpened():

        print(
            f"[ERROR] Could not create output video: {output_path}"
        )

        return

    for path in files:

        writer.write(
            cv2.imread(path)
        )

    writer.release()

    _transcode_to_h264(
        raw_path,
        output_path
    )

    _report_video_write_result(
        output_path,
        "Final video",
        extra=f"{len(files)} frames"
    )


# =============================================================================
# PROCESS VIDEO
# =============================================================================

def process_video(
    video_path,
    style_path,
    output_path,
    model
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            f"[ERROR] Cannot open video: "
            f"{video_path}"
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    original_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    original_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print()
    print("=" * 70)
    print("PROCESSING VIDEO")
    print("=" * 70)

    print(
        f"Content:          "
        f"{os.path.basename(video_path)}"
    )

    print(
        f"Style:            "
        f"{os.path.basename(style_path)}"
    )

    print(
        f"Output:           "
        f"{output_path}"
    )

    print(
        f"FPS:              "
        f"{fps:.2f}"
    )

    print(
        f"Frames:           "
        f"{total_frames}"
    )

    print(
        f"Original size:    "
        f"{original_width} x "
        f"{original_height}"
    )

    # =========================================================================
    # FIRST FRAME
    # =========================================================================

    ret, first_frame = cap.read()

    if not ret:

        cap.release()

        print(
            "[ERROR] Could not read first frame."
        )

        return

    first_frame = resize_frame(
        first_frame
    )

    processing_height, processing_width = (
        first_frame.shape[:2]
    )

    print(
        f"Processing size:  "
        f"{processing_width} x "
        f"{processing_height}"
    )

    # =========================================================================
    # STYLE
    # =========================================================================

    print(
        "Loading style image..."
    )

    style_tensor = load_style(
        style_path
    )

    targets = style_targets(
        model,
        style_tensor
    )

    print(
        "Style features loaded."
    )

    # =========================================================================
    # FRAME CHECKPOINT DIRECTORY (crash / resume support)
    # =========================================================================

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    style_name = os.path.splitext(
        os.path.basename(style_path)
    )[0]

    frames_dir = os.path.join(
        OUTPUT_VIDEO_DIR,
        "_frames",
        f"{video_name}_style_{style_name}"
    )

    os.makedirs(
        frames_dir,
        exist_ok=True
    )

    existing_checkpoints = sorted(
        glob.glob(
            os.path.join(
                frames_dir,
                "frame_*.png"
            )
        )
    )

    resume_count = len(
        existing_checkpoints
    )

    snapshot_dir = (
        os.path.join(
            OUTPUT_VIDEO_DIR,
            "_frame0_progress",
            f"{video_name}_style_{style_name}"
        )
        if SAVE_PROGRESS_SNAPSHOTS
        else None
    )

    # =========================================================================
    # RESET / ADVANCE VIDEO
    # =========================================================================

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    if resume_count > 0:

        print()
        print(
            f"[RESUME] Found {resume_count} already-processed frame(s) "
            f"for this video/style pairing."
        )

        print(
            f"[RESUME] Skipping ahead and continuing from frame "
            f"{resume_count}..."
        )

        for _ in range(resume_count):

            cap.read()

        last_checkpoint = cv2.imread(
            existing_checkpoints[-1]
        )

        previous = frame_to_tensor(
            last_checkpoint
        )

    else:

        previous = None

    # =========================================================================
    # PROCESS FRAMES
    # =========================================================================

    processed_frames = resume_count

    with tqdm(
        total=total_frames,
        initial=resume_count,
        desc="Stylizing video",
        unit="frame"
    ) as progress:

        frame_number = resume_count

        while True:

            if (
                MAX_FRAMES is not None
                and
                frame_number >= MAX_FRAMES
            ):

                print(
                    f"\n[LIMIT] MAX_FRAMES={MAX_FRAMES} reached, "
                    f"stopping early."
                )

                break

            ret, frame = cap.read()

            if not ret:

                break

            frame = resize_frame(
                frame
            )

            content = frame_to_tensor(
                frame
            )

            if frame_number == 0:

                iterations = (
                    FIRST_FRAME_ITERATIONS
                )

                print(
                    "\n[FRAME 0] "
                    "Building strong style anchor..."
                )

            else:

                iterations = (
                    LATER_FRAME_ITERATIONS
                )

            result = stylize_frame(
                content=content,

                model=model,

                style_targets_dict=targets,

                previous=previous,

                iterations=iterations,

                frame_number=frame_number,

                snapshot_dir=(
                    snapshot_dir
                    if frame_number == 0
                    else None
                )
            )

            # -----------------------------------------------------------------
            # Previous-frame initialisation
            # -----------------------------------------------------------------

            previous = (
                result.clone().detach()
            )

            # -----------------------------------------------------------------
            # Convert output to OpenCV frame
            # -----------------------------------------------------------------

            output_frame = tensor_to_frame(
                result
            )

            checkpoint_path = os.path.join(
                frames_dir,
                f"frame_{frame_number:06d}.png"
            )

            cv2.imwrite(
                checkpoint_path,
                output_frame
            )

            processed_frames += 1

            progress.update(
                1
            )

            # -----------------------------------------------------------------
            # Save first-frame comparison
            # -----------------------------------------------------------------

            if frame_number == 0:

                save_debug(
                    frame,
                    output_frame,
                    video_name,
                    style_name
                )

                if SAVE_PROGRESS_SNAPSHOTS:

                    progress_output_path = os.path.join(
                        OUTPUT_VIDEO_DIR,
                        f"{video_name}_style_{style_name}_frame0_progress.mp4"
                    )

                    build_progress_video(
                        snapshot_dir,
                        progress_output_path
                    )

                if DEBUG_FIRST_FRAME_ONLY:

                    print()
                    print(
                        "[DEBUG] "
                        "DEBUG_FIRST_FRAME_ONLY=True"
                    )

                    print(
                        "[DEBUG] "
                        "Only frame 0 was processed."
                    )

                    break

            frame_number += 1

    # =========================================================================
    # RELEASE + ASSEMBLE FINAL VIDEO
    # =========================================================================

    cap.release()

    build_final_video(
        frames_dir,
        output_path,
        fps
    )

    print()
    print(
        "[DONE] Video saved to:"
    )

    print(
        f"       {output_path}"
    )

    print(
        f"       Frames processed this run: "
        f"{processed_frames - resume_count}"
    )

    print(
        f"       Total frames in output:    "
        f"{processed_frames}"
    )


# =============================================================================
# FIND VIDEOS
# =============================================================================

def find_videos():

    patterns = [
        "*.mp4",
        "*.MP4",
        "*.mov",
        "*.MOV"
    ]

    found = []

    seen = set()

    for pattern in patterns:

        for path in glob.glob(
            os.path.join(
                CONTENT_VIDEO_DIR,
                pattern
            )
        ):

            key = os.path.normcase(
                os.path.abspath(path)
            )

            if key not in seen:

                seen.add(key)

                found.append(path)

    return sorted(
        found
    )


# =============================================================================
# FIND STYLE IMAGES
# =============================================================================

def find_styles():

    patterns = [
        "*.jpg",
        "*.JPG",
        "*.jpeg",
        "*.JPEG",
        "*.png",
        "*.PNG",
        "*.webp",
        "*.WEBP"
    ]

    found = []

    seen = set()

    for pattern in patterns:

        for path in glob.glob(
            os.path.join(
                STYLE_IMAGE_DIR,
                pattern
            )
        ):

            key = os.path.normcase(
                os.path.abspath(path)
            )

            if key not in seen:

                seen.add(key)

                found.append(path)

    return sorted(
        found
    )


# =============================================================================
# CONTENT VIDEO SELECTION
# =============================================================================

def choose_video(videos):

    print()
    print("=" * 70)
    print("AVAILABLE CONTENT VIDEOS")
    print("=" * 70)

    for i, path in enumerate(
        videos,
        1
    ):

        print(
            f"  [{i}] "
            f"{os.path.basename(path)}"
        )

    while True:

        choice = input(
            "\nEnter content video number: "
        ).strip()

        try:

            number = int(
                choice
            )

            if 1 <= number <= len(videos):

                selected_video = (
                    videos[number - 1]
                )

                print()
                print(
                    "Selected content video:"
                )

                print(
                    f"  - "
                    f"{os.path.basename(selected_video)}"
                )

                return selected_video

        except ValueError:

            pass

        print(
            "Invalid selection. "
            "Please enter a valid video number."
        )


# =============================================================================
# STYLE SELECTION
# =============================================================================

def choose_styles(styles):

    print()
    print("=" * 70)
    print("AVAILABLE STYLE IMAGES")
    print("=" * 70)

    for i, path in enumerate(
        styles,
        1
    ):

        print(
            f"  [{i}] "
            f"{os.path.basename(path)}"
        )

    while True:

        choice = input(
            "\nEnter style number(s), "
            "e.g. 1 or 1,3, or 'all': "
        ).strip().lower()

        if choice == "all":

            return styles

        try:

            numbers = list(
                dict.fromkeys(
                    int(x.strip())
                    for x in choice.split(",")
                )
            )

            if all(
                1 <= number <= len(styles)
                for number in numbers
            ):

                return [
                    styles[number - 1]
                    for number in numbers
                ]

        except ValueError:

            pass

        print(
            "Invalid selection. Try again."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("VIDEO NEURAL STYLE TRANSFER")
    print("=" * 70)

    print(
        f"Model:                     "
        f"{VGG_MODEL}"
    )

    print(
        "Optimizer:                 LBFGS"
    )

    print(
        f"First frame iterations:    "
        f"{FIRST_FRAME_ITERATIONS}"
    )

    print(
        f"Later frame iterations:    "
        f"{LATER_FRAME_ITERATIONS}"
    )

    print(
        f"Content weight:            "
        f"{CONTENT_WEIGHT:,.0f}"
    )

    print(
        f"Style weight:              "
        f"{STYLE_WEIGHT:,.0f}"
    )

    print(
        f"TV weight:                 "
        f"{TV_WEIGHT}"
    )

    print(
        f"Frame width:               "
        f"{FRAME_WIDTH}"
    )

    print(
        f"Previous-frame init:       "
        f"{USE_PREVIOUS_FRAME}"
    )

    print(
        f"Debug first frame only:    "
        f"{DEBUG_FIRST_FRAME_ONLY}"
    )

    print()

    print(
        f"Content videos:             "
        f"{CONTENT_VIDEO_DIR}"
    )

    print(
        f"Style images:               "
        f"{STYLE_IMAGE_DIR}"
    )

    print(
        f"Output videos:              "
        f"{OUTPUT_VIDEO_DIR}"
    )

    print()

    # =========================================================================
    # DEVICE
    # =========================================================================

    print_device()

    os.makedirs(
        OUTPUT_VIDEO_DIR,
        exist_ok=True
    )

    # =========================================================================
    # FIND INPUTS
    # =========================================================================

    videos = find_videos()

    styles = find_styles()

    print(
        f"Found {len(videos)} "
        f"content video(s)."
    )

    print(
        f"Found {len(styles)} "
        f"style image(s)."
    )

    if not videos:

        print(
            "[ERROR] No content videos found."
        )

        return

    if not styles:

        print(
            "[ERROR] No style images found."
        )

        return

    # =========================================================================
    # SELECT CONTENT VIDEO
    # =========================================================================

    selected_video = choose_video(
        videos
    )

    # =========================================================================
    # SELECT STYLE
    # =========================================================================

    selected_styles = choose_styles(
        styles
    )

    print()
    print(
        "Selected style(s):"
    )

    for style in selected_styles:

        print(
            f"  - "
            f"{os.path.basename(style)}"
        )

    # =========================================================================
    # LOAD VGG16
    # =========================================================================

    print()
    print(
        "Loading VGG16..."
    )

    model = VGG16Features().to(
        DEVICE
    ).eval()

    print(
        "VGG16 loaded."
    )

    # =========================================================================
    # PROCESS SELECTED VIDEO
    # =========================================================================

    video_name = os.path.splitext(
        os.path.basename(selected_video)
    )[0]

    for style in selected_styles:

        style_name = os.path.splitext(
            os.path.basename(style)
        )[0]

        output_path = os.path.join(
            OUTPUT_VIDEO_DIR,

            (
                f"{video_name}"
                f"_style_{style_name}"
                f"_vgg16_lbfgs.mp4"
            )
        )

        process_video(
            video_path=selected_video,

            style_path=style,

            output_path=output_path,

            model=model
        )

    # =========================================================================
    # FINISHED
    # =========================================================================

    print()
    print("=" * 70)
    print("VIDEO NST PROCESSING COMPLETE")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()