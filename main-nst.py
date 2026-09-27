import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable

import numpy as np
import os
import argparse
import csv
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    ssim_metric = None

# =============================================================================
# LOGGING HELPERS
# =============================================================================

def save_iterations_csv(
    log_dir,
    total_losses,
    content_losses,
    style_losses,
    tv_losses,
    content_weight,
    style_weight,
    tv_weight,
    content_weights_per_iter=None,
    style_weights_per_iter=None
):
    csv_path = os.path.join(log_dir, 'iterations.csv')

    adaptive = content_weights_per_iter is not None

    fieldnames = [
        'iteration',
        'total_loss',
        'content_loss (weighted)',
        'style_loss (weighted)',
        'tv_loss (weighted)'
    ]

    if adaptive:
        fieldnames += ['content_weight_used', 'style_weight_used']

    with open(csv_path, 'w', newline='') as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (total_loss, content_loss, style_loss, tv_loss) in enumerate(
            zip(total_losses, content_losses, style_losses, tv_losses), start=1
        ):

            if adaptive:
                current_cw = content_weights_per_iter[i - 1]
                current_sw = style_weights_per_iter[i - 1]
            else:
                current_cw = content_weight
                current_sw = style_weight

            row = {
                'iteration': f'{i:04d}',
                'total_loss': f'{total_loss:.6f}',
                'content_loss (weighted)': f'{content_loss * current_cw:.6f}',
                'style_loss (weighted)': f'{style_loss * current_sw:.6f}',
                'tv_loss (weighted)': f'{tv_loss * tv_weight:.6f}'
            }

            if adaptive:
                row['content_weight_used'] = f'{current_cw:.1f}'
                row['style_weight_used'] = f'{current_sw:.1f}'

            writer.writerow(row)

    print(f'  [LOG] Iteration log saved → {csv_path}')


def save_summary_csv(log_dir, row):
    """
    Save the one-row experiment summary.
    """

    csv_path = os.path.join(log_dir, 'summary.csv')

    fieldnames = [
        'test_name', 'content_image', 'style_image',
        'optimizer', 'model', 'init_method',
        'content_weight', 'style_weight', 'tv_weight',
        'adaptive_weighting', 'time_taken',
        'final_total_loss', 'final_content_loss', 'final_style_loss', 'final_tv_loss',
        'total_loss_min', 'total_loss_max', 'total_loss_mean', 'total_loss_median',
        'content_loss_min', 'content_loss_max', 'content_loss_mean', 'content_loss_median',
        'style_loss_min', 'style_loss_max', 'style_loss_mean', 'style_loss_median',
        'tv_loss_min', 'tv_loss_max', 'tv_loss_mean', 'tv_loss_median',
        'ssim'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f'  [LOG] Summary saved → {csv_path}')


def save_loss_graphs(
    log_dir,
    test_name,
    total_losses,
    content_losses,
    style_losses,
    tv_losses,
    content_weight,
    style_weight,
    tv_weight,
    content_weights_per_iter=None,
    style_weights_per_iter=None
):

    adaptive = content_weights_per_iter is not None

    iterations = list(range(1, len(total_losses) + 1))

    if adaptive:
        weighted_content_losses = [
            loss * weight for loss, weight in zip(content_losses, content_weights_per_iter)
        ]
        weighted_style_losses = [
            loss * weight for loss, weight in zip(style_losses, style_weights_per_iter)
        ]
    else:
        weighted_content_losses = [loss * content_weight for loss in content_losses]
        weighted_style_losses = [loss * style_weight for loss in style_losses]

    weighted_tv_losses = [loss * tv_weight for loss in tv_losses]

    fig = plt.figure(figsize=(18, 12))

    title_suffix = ' [adaptive weights]' if adaptive else ''

    fig.suptitle(f'Loss curves {test_name}{title_suffix}', fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, total_losses, label='Total loss')
    ax1.plot(iterations, weighted_content_losses, label='Weighted content loss', linestyle='--')
    ax1.set_title('Total Loss vs Content Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, total_losses, label='Total loss')
    ax2.plot(iterations, weighted_style_losses, label='Weighted style loss', linestyle='--')
    ax2.set_title('Total Loss vs Style Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(iterations, total_losses, label='Total loss')
    ax3.plot(iterations, weighted_tv_losses, label='Weighted TV loss', linestyle='--')
    ax3.set_title('Total Loss vs TV Loss')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    if adaptive:
        ax3b = ax3.twinx()
        ax3b.plot(iterations, content_weights_per_iter, linestyle=':', label='Content weight')
        ax3b.plot(iterations, style_weights_per_iter, linestyle=':', label='Style weight')
        ax3b.set_ylabel('Weight value')
        ax3b.legend(fontsize=7, loc='upper right')

    ax4 = fig.add_subplot(gs[1, 1])

    metrics = ['Content\n(weighted)', 'Style\n(weighted)', 'TV\n(weighted)', 'Total']
    all_series = [weighted_content_losses, weighted_style_losses, weighted_tv_losses, total_losses]
    statistics = ['min', 'max', 'mean', 'median']

    x = np.arange(len(metrics))
    width = 0.18

    for i, statistic in enumerate(statistics):

        values = []

        for series in all_series:
            array = np.array(series)
            value = np.median(array) if statistic == 'median' else getattr(np, statistic)(array)
            values.append(value)

        ax4.bar(x + i * width, values, width, label=statistic.capitalize())

    ax4.set_title('Min / Max / Mean / Median per Loss')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(metrics, fontsize=8)
    ax4.set_ylabel('Loss value')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    graph_path = os.path.join(log_dir, 'loss_graphs.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  [LOG] Loss graphs saved → {graph_path}')

# =============================================================================
# SSIM EVALUATION
# =============================================================================
def calculate_ssim(content_path, generated_path):

    if ssim_metric is None:
        raise ImportError(
            "scikit-image is required for SSIM. Install it with: pip install scikit-image"
        )

    from PIL import Image

    content = np.asarray(Image.open(content_path).convert('L'), dtype=np.float32)
    generated = np.asarray(Image.open(generated_path).convert('L'), dtype=np.float32)

    # Compare at the same spatial resolution.
    if content.shape != generated.shape:
        from PIL import Image as PILImage
        generated_img = PILImage.open(generated_path).convert('L')
        generated_img = generated_img.resize(
            (content.shape[1], content.shape[0]),
            PILImage.Resampling.LANCZOS
        )
        generated = np.asarray(generated_img, dtype=np.float32)

    score = ssim_metric(content, generated, data_range=255.0)
    return float(score)


def save_ssim_result(log_dir, ssim_value):
    csv_path = os.path.join(log_dir, 'ssim.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ssim'])
        writer.writerow([f'{ssim_value:.6f}'])
    print(f'  [LOG] SSIM saved → {csv_path}')

# =============================================================================
# CLEANUP HELPER
# =============================================================================

def cleanup_test_folder(dump_path, video_ext='.mp4'):

    files = [f for f in os.listdir(dump_path) if os.path.isfile(os.path.join(dump_path, f))]

    video_files = [f for f in files if f.lower().endswith(video_ext)]
    image_files = sorted(f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    if not image_files:
        print('  [!] No image frames found skipping cleanup.')
        return

    final_image = image_files[-1]
    keep = set(video_files) | {final_image}

    for f in files:
        if f not in keep:
            os.remove(os.path.join(dump_path, f))

    print(f'  [CLEANUP] {dump_path} now contains only: {sorted(keep)}')

# =============================================================================
# LOSS FUNCTION
# =============================================================================

def build_loss(
    neural_net,
    optimizing_img,
    target_representations,
    content_feature_maps_index,
    style_feature_maps_indices,
    config,
    current_content_weight=None,
    current_style_weight=None
):

    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]
    current_set_of_feature_maps = neural_net(optimizing_img)
    current_content_representation = (
        current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    )

    content_loss = torch.nn.MSELoss(reduction='mean')(
        target_content_representation, current_content_representation
    )

    current_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(current_set_of_feature_maps)
        if cnt in style_feature_maps_indices
    ]

    style_loss = 0.0

    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])

    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    cw = current_content_weight if current_content_weight is not None else config['content_weight']
    sw = current_style_weight if current_style_weight is not None else config['style_weight']

    total_loss = cw * content_loss + sw * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss

# =============================================================================
# TUNING STEP
# =============================================================================

def make_tuning_step(
    neural_net,
    optimizer,
    target_representations,
    content_feature_maps_index,
    style_feature_maps_indices,
    config
):

    def tuning_step(optimizing_img):

        total_loss, content_loss, style_loss, tv_loss = build_loss(
            neural_net,
            optimizing_img,
            target_representations,
            content_feature_maps_index,
            style_feature_maps_indices,
            config
        )

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step

# =============================================================================
# MAIN NST FUNCTION
# =============================================================================

def neural_style_transfer(config):

    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    content_name = os.path.splitext(os.path.basename(config['content_img_name']))[0]
    style_name = os.path.splitext(os.path.basename(config['style_img_name']))[0]

    base_output_dir = config['output_img_dir']
    os.makedirs(base_output_dir, exist_ok=True)

    test_folder_prefix = f'{content_name}_{style_name}_{config["model"]}_test_'

    test_number = 1

    while True:

        test_name = f'{test_folder_prefix}{test_number:03d}'
        dump_path = os.path.join(base_output_dir, test_name)

        if not os.path.exists(dump_path):
            break

        test_number += 1

    os.makedirs(dump_path, exist_ok=True)

    config['test_name'] = test_name
    config['test_number'] = test_number

    print('\n==================================================')
    print(f'  [OUTPUT] Test folder   : {dump_path}')
    print('==================================================')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_img = utils.prepare_img(style_img_path, config['height'], device)

    if config['init_method'] == 'random':
        gaussian_noise_img = np.random.normal(
            loc=0, scale=90., size=content_img.shape
        ).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        style_img_resized = utils.prepare_img(
            style_img_path, np.asarray(content_img.shape[2:]), device
        )
        init_img = style_img_resized

    optimizing_img = Variable(init_img, requires_grad=True)

    (
        neural_net,
        content_feature_maps_index_name,
        style_feature_maps_indices_names
    ) = utils.prepare_model(config['model'], device)

    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = (
        content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    )

    target_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(style_img_set_of_feature_maps)
        if cnt in style_feature_maps_indices_names[0]
    ]

    target_representations = [target_content_representation, target_style_representation]

    num_of_iterations = {"lbfgs": 1000, "adam": 1000}

    total_losses = []
    content_losses = []
    style_losses = []
    tv_losses = []

    content_weights_used = []
    style_weights_used = []

    adaptive_mode = False
    WARMUP_ITERS = 10

    start_time = time.time()

    # =========================================================================
    # ADAM
    # =========================================================================

    if config['optimizer'] == 'adam':

        adaptive_mode = config.get('adaptive_weighting', False)
        optimizer = Adam((optimizing_img,), lr=1e1)
        n_iters = config.get('num_iterations', num_of_iterations['adam'])

        target_content_contribution = None
        target_style_contribution = None

        for cnt in range(n_iters):

            if adaptive_mode:
                _, content_loss, style_loss, tv_loss = build_loss(
                    neural_net, optimizing_img, target_representations,
                    content_feature_maps_index_name[0],
                    style_feature_maps_indices_names[0],
                    config, current_content_weight=1.0, current_style_weight=1.0
                )

                with torch.no_grad():

                    content_val = content_loss.item()
                    style_val = style_loss.item()
                    eps = 1e-6

                    if cnt < WARMUP_ITERS:
                        cw = config['content_weight']
                        sw = config['style_weight']
                        target_content_contribution = config['content_weight'] * max(content_val, eps)
                        target_style_contribution = config['style_weight'] * max(style_val, eps)

                    else:
                        cw = target_content_contribution / max(content_val, eps)
                        sw = target_style_contribution / max(style_val, eps)

                total_loss = cw * content_loss + sw * style_loss + config['tv_weight'] * tv_loss

            else:
                cw = config['content_weight']
                sw = config['style_weight']

                total_loss, content_loss, style_loss, tv_loss = build_loss(
                    neural_net, optimizing_img, target_representations,
                    content_feature_maps_index_name[0],
                    style_feature_maps_indices_names[0],
                    config, cw, sw)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([optimizing_img], max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                total_losses.append(total_loss.item())
                content_losses.append(content_loss.item())
                style_losses.append(style_loss.item())
                tv_losses.append(tv_loss.item())
                if adaptive_mode:
                    content_weights_used.append(cw)
                    style_weights_used.append(sw)

                if adaptive_mode:
                    print(f'Adam [adaptive] | iter: {cnt+1:04d}, '
                          f'cw={cw:.0f}, sw={sw:.0f}, '
                          f'total={total_loss.item():12.4f}, '
                          f'content={cw * content_loss.item():12.4f}, '
                          f'style={sw * style_loss.item():12.4f}')
                else:
                    print(f'Adam | iteration: {cnt+1:04d}, '
                          f'total loss={total_loss.item():12.4f}, '
                          f'content_loss={config["content_weight"] * content_loss.item():12.4f}, '
                          f'style loss={config["style_weight"] * style_loss.item():12.4f}, '
                          f'tv loss={config["tv_weight"] * tv_loss.item():12.4f}')

                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt,
                                             n_iters, should_display=False)

    # =========================================================================
    # L-BFGS
    # =========================================================================

    elif config['optimizer'] == 'lbfgs':

        adaptive_mode = config.get('adaptive_weighting', False)
        n_iters = config.get('num_iterations', num_of_iterations['lbfgs'])

        target_content_contribution = None
        target_style_contribution = None

        # Held fixed for the duration of each optimizer.step() call.
        current_cw = config['content_weight']
        current_sw = config['style_weight']

        # Smooths cw/sw updates so the objective stays close to stationary
        # between L-BFGS steps (large swings break its curvature memory).
        EMA_ALPHA = 0.9

        # Rebuild the optimizer if the loss stalls for this many
        # consecutive iterations (stale curvature memory vs. the
        # still-shifting adaptive objective).
        STAGNATION_PATIENCE = 3
        STAGNATION_TOL = 1e-6
        stagnant_count = 0
        prev_total_loss = None

        optimizer = LBFGS(
            (optimizing_img,),
            max_iter=1,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn='strong_wolfe'
        )

        def closure():

            if torch.is_grad_enabled():
                optimizer.zero_grad()

            total_loss, content_loss, style_loss, tv_loss = build_loss(
                neural_net, optimizing_img, target_representations,
                content_feature_maps_index_name[0],
                style_feature_maps_indices_names[0],
                config, current_content_weight=current_cw, current_style_weight=current_sw)

            if total_loss.requires_grad:
                total_loss.backward()

            return total_loss

        for cnt in range(n_iters):

            if adaptive_mode:

                # Probe raw loss magnitudes before this step (image held
                # fixed) so cw/sw are set from a stationary point.
                with torch.no_grad():
                    _, content_loss_probe, style_loss_probe, _ = build_loss(
                        neural_net, optimizing_img, target_representations,
                        content_feature_maps_index_name[0],
                        style_feature_maps_indices_names[0],
                        config, current_content_weight=1.0, current_style_weight=1.0)

                    content_val = content_loss_probe.item()
                    style_val = style_loss_probe.item()
                    eps = 1e-6

                    if cnt < WARMUP_ITERS:
                        current_cw = config['content_weight']
                        current_sw = config['style_weight']

                        target_content_contribution = config['content_weight'] * max(content_val, eps)
                        target_style_contribution = config['style_weight'] * max(style_val, eps)

                    else:
                        raw_cw = target_content_contribution / max(content_val, eps)
                        raw_sw = target_style_contribution / max(style_val, eps)

                        current_cw = EMA_ALPHA * current_cw + (1 - EMA_ALPHA) * raw_cw
                        current_sw = EMA_ALPHA * current_sw + (1 - EMA_ALPHA) * raw_sw

            else:
                current_cw = config['content_weight']
                current_sw = config['style_weight']

            optimizer.step(closure)

            # Only relevant in adaptive mode a fixed-weight objective
            # going flat just means converged, not stuck.
            if adaptive_mode:

                with torch.no_grad():
                    probe_total_loss, _, _, _ = build_loss(
                        neural_net, optimizing_img, target_representations,
                        content_feature_maps_index_name[0],
                        style_feature_maps_indices_names[0],
                        config, current_content_weight=current_cw, current_style_weight=current_sw)

                    probe_val = probe_total_loss.item()

                if prev_total_loss is not None and abs(probe_val - prev_total_loss) < STAGNATION_TOL:
                    stagnant_count += 1
                else:
                    stagnant_count = 0

                prev_total_loss = probe_val

                if stagnant_count >= STAGNATION_PATIENCE:
                    print(f'  [!] L-BFGS stagnation detected at iter {cnt+1:04d} '
                          f'rebuilding optimizer to clear stale curvature memory.')

                    optimizer = LBFGS(
                        (optimizing_img,),
                        max_iter=1,
                        tolerance_grad=1e-10,
                        tolerance_change=1e-12,
                        line_search_fn='strong_wolfe'
                    )
                    stagnant_count = 0

            with torch.no_grad():
                total_loss, content_loss, style_loss, tv_loss = build_loss(
                    neural_net, optimizing_img, target_representations,
                    content_feature_maps_index_name[0],
                    style_feature_maps_indices_names[0],
                    config, current_content_weight=current_cw, current_style_weight=current_sw)

                total_losses.append(total_loss.item())
                content_losses.append(content_loss.item())
                style_losses.append(style_loss.item())
                tv_losses.append(tv_loss.item())

                if adaptive_mode:
                    content_weights_used.append(current_cw)
                    style_weights_used.append(current_sw)

                    print(f'L-BFGS [adaptive] | iter: {cnt+1:04d}, '
                          f'cw={current_cw:.0f}, sw={current_sw:.0f}, '
                          f'total={total_loss.item():12.4f}, '
                          f'content={current_cw * content_loss.item():12.4f}, '
                          f'style={current_sw * style_loss.item():12.4f}')
                else:
                    print(f'L-BFGS | iteration: {cnt+1:04d}, total loss={total_loss.item():12.4f}, '
                          f'content_loss={config["content_weight"] * content_loss.item():12.4f}, '
                          f'style loss={config["style_weight"] * style_loss.item():12.4f}, '
                          f'tv loss={config["tv_weight"] * tv_loss.item():12.4f}')

                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt,
                                             n_iters, should_display=False)

    # =========================================================================
    # FINISHED
    # =========================================================================

    elapsed = time.time() - start_time
    mins, sec = divmod(int(elapsed), 60)
    time_str = f'{mins:02d}:{sec:02d}'

    summary_test_name = f'Style {style_name} and Content {content_name}'

    # =========================================================================
    # LOG DIRECTORY
    # =========================================================================

    base_folder = f'{content_name}_{style_name}_{config["model"]}'

    data_visual = os.path.join(os.path.dirname(__file__), 'data', 'data-visual')
    log_dir = os.path.join(data_visual, base_folder, config['test_name'])

    os.makedirs(log_dir, exist_ok=True)

    print(f'\n  [LOG] Saving results to → {log_dir}')

    if adaptive_mode:
        effective_content_weight = content_weights_used[-1]
        effective_style_weight = style_weights_used[-1]
    else:
        effective_content_weight = config['content_weight']
        effective_style_weight = config['style_weight']

    def _stats(values):
        array = np.array(values)
        return array.min(), array.max(), array.mean(), np.median(array)

    total_min, total_max, total_mean, total_median = _stats(total_losses)
    content_min, content_max, content_mean, content_median = _stats(
        [value * effective_content_weight for value in content_losses]
    )
    style_min, style_max, style_mean, style_median = _stats(
        [value * effective_style_weight for value in style_losses]
    )
    tv_min, tv_max, tv_mean, tv_median = _stats(
        [value * config['tv_weight'] for value in tv_losses]
    )

    # -------------------------------------------------------------------------
    # SSIM
    # -------------------------------------------------------------------------

    final_image_name = f'{config["test_name"]}{config["img_format"][1]}'
    final_image_path = os.path.join(dump_path, final_image_name)

    ssim_value = calculate_ssim(content_img_path, final_image_path)

    print(f'  [SSIM] Content preservation SSIM: {ssim_value:.6f}')
    save_ssim_result(log_dir, ssim_value)

    save_iterations_csv(
        log_dir,
        total_losses, content_losses, style_losses, tv_losses,
        config['content_weight'], config['style_weight'], config['tv_weight'],
        content_weights_per_iter=content_weights_used if adaptive_mode else None,
        style_weights_per_iter=style_weights_used if adaptive_mode else None
    )

    save_loss_graphs(
        log_dir,
        summary_test_name,
        total_losses, content_losses, style_losses, tv_losses,
        config['content_weight'], config['style_weight'], config['tv_weight'],
        content_weights_per_iter=content_weights_used if adaptive_mode else None,
        style_weights_per_iter=style_weights_used if adaptive_mode else None
    )

    save_summary_csv(
        log_dir,
        {
            'test_name': summary_test_name,
            'content_image': config['content_img_name'],
            'style_image': config['style_img_name'],
            'optimizer': config['optimizer'],
            'model': config['model'],
            'init_method': config['init_method'],
            'content_weight': config['content_weight'],
            'style_weight': config['style_weight'],
            'tv_weight': config['tv_weight'],
            'adaptive_weighting': 'yes' if adaptive_mode else 'no',
            'time_taken': time_str,
            'final_total_loss': f'{total_losses[-1]:.6f}',
            'final_content_loss': f'{content_losses[-1]:.6f}',
            'final_style_loss': f'{style_losses[-1]:.6f}',
            'final_tv_loss': f'{tv_losses[-1]:.6f}',
            'total_loss_min': f'{total_min:.6f}',
            'total_loss_max': f'{total_max:.6f}',
            'total_loss_mean': f'{total_mean:.6f}',
            'total_loss_median': f'{total_median:.6f}',
            'content_loss_min': f'{content_min:.6f}',
            'content_loss_max': f'{content_max:.6f}',
            'content_loss_mean': f'{content_mean:.6f}',
            'content_loss_median': f'{content_median:.6f}',
            'style_loss_min': f'{style_min:.6f}',
            'style_loss_max': f'{style_max:.6f}',
            'style_loss_mean': f'{style_mean:.6f}',
            'style_loss_median': f'{style_median:.6f}',
            'tv_loss_min': f'{tv_min:.6f}',
            'tv_loss_max': f'{tv_max:.6f}',
            'tv_loss_mean': f'{tv_mean:.6f}',
            'tv_loss_median': f'{tv_median:.6f}',
            'ssim': f'{ssim_value:.6f}'
        }
    )

    print(f'  [OUTPUT] Final image → {final_image_path}')

    return dump_path

# =============================================================================
# HYPERPARAMETER GRID SEARCH
# =============================================================================

def parse_float_list(value):
    """Parse comma-separated numeric values for grid-search arguments."""
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def run_grid_search(base_config):
    """
    Run a Cartesian-product grid over content/style/TV weights.

    Each configuration is a normal NST experiment, so every result keeps
    its own final image, loss logs, summary.csv and SSIM score.
    """
    content_weights = parse_float_list(base_config['grid_content_weights'])
    style_weights = parse_float_list(base_config['grid_style_weights'])
    tv_weights = parse_float_list(base_config['grid_tv_weights'])

    results = []
    total = len(content_weights) * len(style_weights) * len(tv_weights)

    print('\n==================================================')
    print('  HYPERPARAMETER GRID SEARCH')
    print('==================================================')
    print(f'  Content weights : {content_weights}')
    print(f'  Style weights   : {style_weights}')
    print(f'  TV weights      : {tv_weights}')
    print(f'  Experiments     : {total}')
    print(f'  Iterations/test : {base_config["num_iterations"]}')

    counter = 0

    for cw in content_weights:
        for sw in style_weights:
            for tv in tv_weights:
                counter += 1
                config = dict(base_config)
                config['content_weight'] = cw
                config['style_weight'] = sw
                config['tv_weight'] = tv
                config['adaptive_weighting'] = False
                config['grid_search_mode'] = True

                print('\n--------------------------------------------------')
                print(f'  GRID {counter}/{total}')
                print(f'  content_weight={cw}, style_weight={sw}, tv_weight={tv}')
                print('--------------------------------------------------')

                dump_path = neural_style_transfer(config)

                summary_path = None
                for root, _, files in os.walk(dump_path):
                    if 'summary.csv' in files:
                        summary_path = os.path.join(root, 'summary.csv')
                        break

                row = {
                    'experiment': counter,
                    'content_weight': cw,
                    'style_weight': sw,
                    'tv_weight': tv,
                    'result_path': dump_path
                }

                if summary_path:
                    with open(summary_path, newline='') as f:
                        data = next(csv.DictReader(f))
                    row['final_total_loss'] = data.get('final_total_loss', '')
                    row['final_content_loss'] = data.get('final_content_loss', '')
                    row['final_style_loss'] = data.get('final_style_loss', '')
                    row['ssim'] = data.get('ssim', '')

                results.append(row)

    search_dir = os.path.join(
        os.path.dirname(__file__), 'data', 'data-visual', 'grid-search'
    )
    os.makedirs(search_dir, exist_ok=True)
    csv_path = os.path.join(search_dir, 'grid_search_results.csv')

    fields = ['experiment', 'content_weight', 'style_weight', 'tv_weight',
              'final_total_loss', 'final_content_loss', 'final_style_loss',
              'ssim', 'result_path']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    # Higher SSIM is preferred for structural preservation.
    valid = [r for r in results if r.get('ssim', '') not in ('', None)]
    if valid:
        best = max(valid, key=lambda r: float(r['ssim']))
        print('\n==================================================')
        print('  GRID SEARCH COMPLETE')
        print('==================================================')
        print(f'  Best SSIM       : {float(best["ssim"]):.6f}')
        print(f'  Content weight  : {best["content_weight"]}')
        print(f'  Style weight    : {best["style_weight"]}')
        print(f'  TV weight       : {best["tv_weight"]}')
        print(f'  Results CSV     : {csv_path}')

    return results

# =============================================================================
# ABLATION STUDY
# =============================================================================

def run_ablation_study(base_config):
    """
    Compare the complete NST objective against versions with one loss
    component removed at a time.

    Variants:
      full       = content + style + TV
      no_content = style + TV
      no_style   = content + TV
      no_tv      = content + style
    """
    variants = [
        ('full', base_config['content_weight'], base_config['style_weight'], base_config['tv_weight']),
        ('no_content', 0.0, base_config['style_weight'], base_config['tv_weight']),
        ('no_style', base_config['content_weight'], 0.0, base_config['tv_weight']),
        ('no_tv', base_config['content_weight'], base_config['style_weight'], 0.0),
    ]

    results = []

    print('\n==================================================')
    print('  ABLATION STUDY')
    print('==================================================')
    print('  full       : content + style + TV')
    print('  no_content : style + TV')
    print('  no_style   : content + TV')
    print('  no_tv      : content + style')
    print(f'  Iterations/test : {base_config["num_iterations"]}')

    for variant, cw, sw, tv in variants:
        config = dict(base_config)
        config['content_weight'] = cw
        config['style_weight'] = sw
        config['tv_weight'] = tv
        config['adaptive_weighting'] = False
        config['ablation_mode'] = True

        print('\n--------------------------------------------------')
        print(f'  ABLATION: {variant}')
        print('--------------------------------------------------')

        dump_path = neural_style_transfer(config)

        summary_path = None
        for root, _, files in os.walk(dump_path):
            if 'summary.csv' in files:
                summary_path = os.path.join(root, 'summary.csv')
                break

        row = {
            'variant': variant,
            'content_weight': cw,
            'style_weight': sw,
            'tv_weight': tv,
            'result_path': dump_path
        }

        if summary_path:
            with open(summary_path, newline='') as f:
                data = next(csv.DictReader(f))
            row['final_total_loss'] = data.get('final_total_loss', '')
            row['final_content_loss'] = data.get('final_content_loss', '')
            row['final_style_loss'] = data.get('final_style_loss', '')
            row['final_tv_loss'] = data.get('final_tv_loss', '')
            row['ssim'] = data.get('ssim', '')

        results.append(row)

    ablation_dir = os.path.join(
        os.path.dirname(__file__), 'data', 'data-visual', 'ablation-study'
    )
    os.makedirs(ablation_dir, exist_ok=True)
    csv_path = os.path.join(ablation_dir, 'ablation_results.csv')

    fields = ['variant', 'content_weight', 'style_weight', 'tv_weight',
              'final_total_loss', 'final_content_loss', 'final_style_loss',
              'final_tv_loss', 'ssim', 'result_path']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print('\n==================================================')
    print('  ABLATION STUDY COMPLETE')
    print(f'  Results CSV: {csv_path}')
    print('==================================================')

    return results

# =============================================================================
# IMAGE SELECTION HELPERS
# =============================================================================

def list_images(directory, label):

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')

    if not os.path.exists(directory):
        print(f'  [!] Directory not found: {directory}')
        return []

    files = sorted([
        filename for filename in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, filename))
        and filename.lower().endswith(supported_exts)
    ])

    if not files:
        print(f'  [!] No images found in {directory}')
        return []

    print(f'\n  Available {label} images:')

    for i, filename in enumerate(files, 1):
        print(f'    {i}. {filename}')

    return files


def pick_image(files, label):

    while True:

        try:
            choice = int(input(f'\n  Enter the number for your {label} image: '))

            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print(f'  Please enter a number between 1 and {len(files)}.')

        except ValueError:
            print('  Invalid input. Please enter a number.')


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')

    img_format = (4, '.jpg')

    parser = argparse.ArgumentParser()

    parser.add_argument('--content_img_name', type=str, default=None)
    parser.add_argument('--style_img_name', type=str, default=None)
    parser.add_argument('--height', type=int, default=400)
    parser.add_argument('--content_weight', type=float, default=1e5)
    parser.add_argument('--style_weight', type=float, default=3e4)
    parser.add_argument('--tv_weight', type=float, default=1e0)
    parser.add_argument('--optimizer', type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument('--model', type=str, choices=['vgg16', 'vgg19'], default='vgg16')
    parser.add_argument('--init_method', type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument('--saving_freq', type=int, default=-1)

    # -------------------------------------------------------------------------
    # Experiment modes
    # -------------------------------------------------------------------------
    parser.add_argument('--grid_search', action='store_true',
                        help='Run a Cartesian hyperparameter grid search.')
    parser.add_argument('--ablation_study', action='store_true',
                        help='Run full/no-content/no-style/no-TV ablation study.')
    parser.add_argument('--grid_content_weights', type=str, default='50000,100000')
    parser.add_argument('--grid_style_weights', type=str, default='30000,60000')
    parser.add_argument('--grid_tv_weights', type=str, default='0.1,1')
    parser.add_argument('--grid_iterations', type=int, default=300,
                        help='Iterations used for each grid/ablation experiment.')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Iterations for a normal NST run.')
    parser.add_argument('--skip_video', action='store_true',
                        help='Do not create the intermediate-results video.')
    parser.add_argument(
        '--adaptive_weighting',
        action='store_true',
        help='Dynamically rebalance content/style weights based on live loss magnitudes '
             '(after a short fixed-weight warm-up).'
    )

    args = parser.parse_args()

    if args.content_img_name is None:

        content_files = list_images(content_images_dir, 'content')

        if not content_files:
            raise SystemExit('No content images found.')

        args.content_img_name = pick_image(content_files, 'content')

    if args.style_img_name is None:

        style_files = list_images(style_images_dir, 'style')

        if not style_files:
            raise SystemExit('No style images found.')

        args.style_img_name = pick_image(style_files, 'style')

    print(f'\n  Content image      : {args.content_img_name}')
    print(f'  Style image        : {args.style_img_name}')
    print(f'  Optimizer          : {args.optimizer}')
    print(f'  Model              : {args.model}')
    print(f'  Adaptive weighting : {args.adaptive_weighting}')
    print(f'  Iterations         : {args.num_iterations}')

    optimization_config = {arg: getattr(args, arg) for arg in vars(args)}
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    # -------------------------------------------------------------------------
    # Experiment modes
    # -------------------------------------------------------------------------
    if args.grid_search and args.ablation_study:
        raise SystemExit('Choose only one: --grid_search or --ablation_study.')

    if args.grid_search:
        optimization_config['num_iterations'] = args.grid_iterations
        run_grid_search(optimization_config)

    elif args.ablation_study:
        optimization_config['num_iterations'] = args.grid_iterations
        run_ablation_study(optimization_config)

    else:
        results_path = neural_style_transfer(optimization_config)

        if not args.skip_video:
            create_video_from_intermediate_results(results_path, img_format)

        cleanup_test_folder(results_path)

        print('\n==================================================')
        print('NST COMPLETE')
        print(f'All results saved in:\n{results_path}')
        print('==================================================')