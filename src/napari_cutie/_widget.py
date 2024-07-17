from typing import TYPE_CHECKING

import tifffile
from magicgui import magic_factory
import numpy as np
import cv2
import tifffile as tiff
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


if TYPE_CHECKING:
    import napari


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
def track_with_cutie(img_stack, mask, cell_dist=4, padding=50):
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
    mask_contours = {i: find_contours(mask == i) for i in mask_numbers}
    batches = find_nearest_masks(mask_contours, cell_dist)

    combined_mask = np.zeros_like(img_stack[1:], dtype=np.uint8)
    print(f'batch number: {len(batches)}')

    # for batch in batches:
    # add tqdm
    for batch in tqdm(batches, desc='Tracking batch'):
        processor = InferenceCore(cutie, cfg=cutie.cfg)
        batch_mask = np.zeros_like(mask, dtype=np.uint8)
        for mask_number in batch:
            batch_mask += (mask == mask_number).astype(np.uint8) * mask_number

        x, y, w, h = cv2.boundingRect(batch_mask)
        pad_x1 = max(x - padding, 0)
        pad_y1 = max(y - padding, 0)
        pad_x2 = min(x + w + padding, mask.shape[1])
        pad_y2 = min(y + h + padding, mask.shape[0])
        cropped_images = [img[pad_y1:pad_y2, pad_x1:pad_x2] for img in tiff_stack_rgb[1:]]
        template_mask = batch_mask[pad_y1:pad_y2, pad_x1:pad_x2]

        # plt.figure(figsize=(5, 5))
        # plt.imshow(cropped_images[0])
        # plt.imshow(template_mask, alpha=0.5)
        # plt.title("Template mask with cropped image")
        # plt.show()

        # Convert template_mask to a tensor and move it to the same device as the image
        template_mask_tensor = torch.from_numpy(template_mask).cuda()

        batch_int = [int(b) for b in batch]  # Convert batch elements to integers
        # make masks frames - 1, because first mask is template, no need to process it
        masks = np.zeros_like(cropped_images, dtype=np.uint8)
        # make masks without rgb channels
        masks = masks[:, :, :, 0]

        for ti, image in enumerate(cropped_images):
            image_pil = Image.fromarray(image)
            image_tensor = to_tensor(image_pil).cuda().float()
            if ti == 0:
                output_prob = processor.step(image_tensor, template_mask_tensor, objects=batch_int, idx_mask=True)
            else:
                output_prob = processor.step(image_tensor, idx_mask=False)

            # mask = processor.output_prob_to_mask(output_prob)
            template_mask_tensor = processor.output_prob_to_mask(output_prob)
            # mask_pil = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
            masks[ti] = template_mask_tensor.cpu().numpy().astype(np.uint8)
            mask_pil = Image.fromarray(masks[ti])
            # mask_pil.show()
            # show the generated mask
            # plt.figure(figsize=(5, 5))
            # plt.imshow(image)
            # plt.imshow(mask_pil, alpha=0.5)
            # plt.title(f"Generated mask at frame {ti}")
            # plt.show()

        combined_mask[:, pad_y1:pad_y2, pad_x1:pad_x2] = np.maximum(combined_mask[:, pad_y1:pad_y2, pad_x1:pad_x2], masks)

    return combined_mask


@magic_factory(call_button='track')
def cutie_track_widget(
    img_stack: "napari.layers.Image", annotation: "napari.layers.Labels",
    cell_dist: "int" = 4, padding: "int" = 20
) -> "napari.types.LabelsData":
    annotation = annotation.data[0]
    if np.max(annotation) == 0:
        raise NotImplementedError
    return track_with_cutie(img_stack.data, annotation, cell_dist, padding)
