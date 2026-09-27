# utils/utils.py

import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from models.definitions.vgg_nets import Vgg16, Vgg16Experimental, Vgg19


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


#
# Image manipulation util functions
#

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')

    with open(img_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    img = cv.imdecode(raw, cv.IMREAD_COLOR)

    if img is None:
        raise Exception(
            f'OpenCV could not decode image '
            f'(unsupported format or corrupt file): {img_path}'
        )

    img = img[:, :, ::-1]

    if target_shape is not None:

        if isinstance(target_shape, int) and target_shape != -1:

            current_height, current_width = img.shape[:2]

            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))

            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)

        else:

            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32)
    img /= 255.0

    return img


def prepare_img(img_path, target_shape, device):

    img = load_image(img_path, target_shape=target_shape)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device).unsqueeze(0)

    return img


def save_image(img, img_path):

    if isinstance(img, torch.Tensor):

        img = img.detach().cpu().numpy()

        if img.ndim == 4:
            img = img.squeeze(0)

        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = np.clip(img, 0, 255).astype(np.uint8)

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    cv.imwrite(img_path, img)


#
# Output naming
#

def generate_out_img_name(config):
    """
    Build the final image filename directly from the test_name that
    neural_style_transfer.py already computed for the dump folder, so
    the saved file always matches the folder it's saved into
    (content_style_test_XXX).
    """

    return f'{config["test_name"]}{config["img_format"][1]}'


def save_and_maybe_display(
    optimizing_img,
    dump_path,
    config,
    img_id,
    num_of_iterations,
    should_display=False
):

    saving_freq = config['saving_freq']

    out_img = (
        optimizing_img
        .squeeze(axis=0)
        .to('cpu')
        .detach()
        .numpy()
    )

    out_img = np.moveaxis(out_img, 0, 2)

    # ---------------------------------------------------------
    # Save only final image when saving_freq == -1
    #
    # Otherwise:
    # - save according to saving_freq
    # - always save final image
    # ---------------------------------------------------------

    if (
        img_id == num_of_iterations - 1
        or (saving_freq > 0 and img_id % saving_freq == 0)
    ):

        img_format = config['img_format']

        if saving_freq != -1:
            out_img_name = str(img_id).zfill(img_format[0]) + img_format[1]
        else:
            out_img_name = generate_out_img_name(config)

        dump_img = np.copy(out_img)

        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))

        dump_img = np.clip(dump_img, 0, 255).astype(np.uint8)

        dump_img = cv.cvtColor(dump_img, cv.COLOR_RGB2BGR)

        output_path = os.path.join(dump_path, out_img_name)

        os.makedirs(dump_path, exist_ok=True)

        cv.imwrite(output_path, dump_img)

        print(f'  [OUTPUT] Final image saved → {output_path}')

    if should_display:

        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()


def get_uint8_range(x):

    if isinstance(x, np.ndarray):

        x = x.copy()
        x -= np.min(x)

        max_value = np.max(x)

        if max_value != 0:
            x /= max_value

        x *= 255

        return x

    else:
        raise ValueError(f'Expected numpy array got {type(x)}')


#
# Model preparation
#

def prepare_model(model, device):

    experimental = False

    if model == 'vgg16':

        if experimental:
            model = Vgg16Experimental(requires_grad=False, show_progress=True)
        else:
            model = Vgg16(requires_grad=False, show_progress=True)

    elif model == 'vgg19':
        model = Vgg19(requires_grad=False, show_progress=True)

    else:
        raise ValueError(f'{model} not supported.')

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)

    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names


#
# Gram matrix
#

def gram_matrix(x, should_normalize=True):

    (b, ch, h, w) = x.size()

    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)

    gram = features.bmm(features_t)

    if should_normalize:
        gram /= (ch * h * w)

    return gram


#
# Total variation
#

def total_variation(y):

    return (
        torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
        +
        torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    )