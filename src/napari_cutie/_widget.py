from typing import TYPE_CHECKING
import re
from magicgui import magic_factory
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm, trange
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from enum import Enum
import time


if TYPE_CHECKING:
    import napari


class ModeChoice(Enum):
    fixed_field = 'fixed'
    shift_field = 'shift'
    atomic = 'atomic'


def find_contours(mask):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.squeeze(contour, axis=1) for contour in contours]
    for i, contour in enumerate(contours):
        if not np.array_equal(contour[0], contour[-1]):
            contours[i] = np.vstack([contour, contour[0]])
    contours = np.array(contours, dtype=np.uint16)
    return contours


def find_nearest_masks(mask_contours, cell_dist):
    mask_numbers = list(mask_contours.keys())
    mask_numbers = np.array(mask_numbers, dtype=np.uint8)
    batches = []
    processed = set()

    for index in mask_numbers:
        if index in processed:
            continue
        current_batch = {index}
        stack = [index]
        while stack:
            current_index = stack.pop()
            for other_index in mask_numbers:
                if other_index not in processed and other_index != current_index:
                    if len(mask_contours[current_index]) > 0 and len(mask_contours[other_index]) > 0:
                        min_dist = np.inf
                        for contour1 in mask_contours[current_index]:
                            for contour2 in mask_contours[other_index]:
                                dist = cdist(contour1, contour2, 'euclidean').min()
                                if dist < min_dist:
                                    min_dist = dist
                        if min_dist < cell_dist:
                            if other_index not in current_batch:
                                stack.append(other_index)
                                current_batch.add(other_index)
                                processed.add(other_index)
        batches.append(list(current_batch))

    return batches


@torch.inference_mode()
@torch.cuda.amp.autocast()
def track_with_cutie(img_stack, mask, cell_dist=4, padding=50, mode=ModeChoice.fixed_field,
                     frame=0, cutie=None, recur=False):
    if not cutie:
        cutie = get_default_model()

    log_scale = 0.005
    log_constant = 65535 / np.log(1 + 65535 * log_scale)

    colormap = plt.get_cmap('viridis')
    tiff_stack_rgb = np.zeros((img_stack.shape[0], img_stack.shape[1], img_stack.shape[2], 3), dtype=np.uint8)
    for i, img in enumerate(img_stack):
        img = img.astype(np.float32)
        img = img - np.min(img)
        img = np.log(1 + img * log_scale) * log_constant
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        tiff_rgb = colormap(img)[:, :, :3]
        tiff_rgb = (tiff_rgb * 255).astype(np.uint8)
        tiff_stack_rgb[i] = tiff_rgb

    mask_numbers = np.unique(mask)
    mask_numbers = mask_numbers[mask_numbers != 0]
    if recur:
        batches = [mask_numbers]
    else:
        mask_contours = {i: find_contours(mask == i) for i in mask_numbers}
        batches = find_nearest_masks(mask_contours, cell_dist)

    combined_mask = np.zeros_like(img_stack[0:], dtype=np.uint8)
    combined_mask[0] = mask

    if mode == ModeChoice.fixed_field or mode == ModeChoice.shift_field:
        start_time = time.time()
        for batch in tqdm(batches, desc='Tracking batch', leave=False):
            processor = InferenceCore(cutie, cfg=cutie.cfg)
            batch_mask = np.zeros_like(mask, dtype=np.uint8)
            for mask_number in batch:
                batch_mask += (mask == mask_number).astype(np.uint8) * mask_number

            x, y, w, h = cv2.boundingRect(batch_mask)
            pad_x1 = max(x - padding, 0)
            pad_y1 = max(y - padding, 0)
            pad_x2 = min(x + w + padding, mask.shape[1])
            pad_y2 = min(y + h + padding, mask.shape[0])
            dif_x = pad_x2 - pad_x1
            dif_y = pad_y2 - pad_y1
            template_mask = batch_mask[pad_y1:pad_y2, pad_x1:pad_x2]

            # Convert template_mask to a tensor and move it to the same device as the image
            template_mask_tensor = torch.from_numpy(template_mask).cuda()

            batch_int = [int(b) for b in batch]  # Convert batch elements to integers
            # make masks frames - 1, because first mask is template, no need to process it
            masks = np.zeros_like(tiff_stack_rgb, dtype=np.uint8)
            # make masks without rgb channels
            masks = masks[:, :, :, 0]

            for ti in range(tiff_stack_rgb.shape[0]):
                image = tiff_stack_rgb[ti, pad_y1:pad_y2, pad_x1:pad_x2, :]
                image_pil = Image.fromarray(image)
                image_tensor = to_tensor(image_pil).cuda().float()
                if ti == 0:
                    output_prob = processor.step(image_tensor, template_mask_tensor, objects=batch_int, idx_mask=True)
                else:
                    output_prob = processor.step(image_tensor, idx_mask=False)

                template_mask_tensor = processor.output_prob_to_mask(output_prob)
                mask_this_frame = template_mask_tensor.cpu().numpy().astype(np.uint8)
                masks[ti, pad_y1:pad_y2, pad_x1:pad_x2] = mask_this_frame
                if mode == ModeChoice.shift_field:
                    x, y, _, _ = cv2.boundingRect(masks[ti])
                    pad_x1 = max(x - padding, 0)
                    pad_y1 = max(y - padding, 0)
                    pad_x2 = min(pad_x1 + dif_x, mask.shape[1])
                    if pad_x2 == mask.shape[1]:
                        pad_x1 = pad_x2 - dif_x
                    pad_y2 = min(pad_y1 + dif_y, mask.shape[0])
                    if pad_y2 == mask.shape[0]:
                        pad_y1 = pad_y2 - dif_y

            combined_mask[:, :, :] = np.maximum(combined_mask[:, :, :], masks)
        name = f'{mode.value}_track_result'
        end_time = time.time()
        duration = end_time - start_time
        print(f'time: {duration}')

        return combined_mask, {"name": name}, "labels"

    elif mode == ModeChoice.atomic:
        start_time = time.time()
        for ti in trange(tiff_stack_rgb.shape[0] - 1):
            atomic_combined_mask, _, _ = track_with_cutie(img_stack[ti: ti + 2], combined_mask[ti], cell_dist,
                                                          padding, ModeChoice.fixed_field, ti, cutie, recur=True)
            combined_mask[ti + 1] = atomic_combined_mask[1]
        end_time = time.time()
        duration = end_time - start_time
        print(f'time: {duration}')
        name = f'{mode.value}_track_result'
        return combined_mask, {"name": name}, "labels"

    else:
        raise NotImplementedError


@magic_factory(call_button='track')
def cutie_track_widget(
    img_stack: "napari.layers.Image", annotation: "napari.layers.Labels",
    cell_dist: "int" = 4, padding: "int" = 20, mode: ModeChoice = ModeChoice.fixed_field
) -> "napari.types.LayerDataTuple":
    # 3d labels layer
    if len(annotation.data.shape) == 3:
        annotation = annotation.data[0]
        if np.max(annotation) == 0:
            raise NotImplementedError
        frame = 0
    # 2d labels layer
    elif len(annotation.data.shape) == 2:
        if np.max(annotation.data) == 0:
            raise NotImplementedError
        match = re.search(r'_(\d+)', annotation.name)
        annotation = annotation.data
        frame = int(match.groups()[0])
    # error
    else:
        raise ValueError
    return track_with_cutie(img_stack.data, annotation, cell_dist, padding, mode, frame)
