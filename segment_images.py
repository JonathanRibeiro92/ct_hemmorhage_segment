from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def resize_img(img, img_shape: Tuple[int, int] = (512, 512)):
    img = img.astype(np.uint8)
    img = cv2.resize(img, img_shape)
    return img


def generate_histogram(img, brain_mask=None):
    img = img.astype(np.uint8)
    histogram = cv2.calcHist([img], [0], brain_mask, [512], [0, 512])
    return histogram


def apply_mask_hide_pixel(img, min=30, max=230):
    mask = np.zeros_like(img) + 255
    mask[img > max] = 0
    mask[img < min] = 0
    return mask


def plot_big(img, cmap="Greys_r"):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=cmap)


def erode_dilate_img(img, mask):
    masked_img = np.copy(img)
    masked_img[mask == 0] = 0
    masked_img = cv2.erode(masked_img, np.ones((3, 3)))
    masked_img = cv2.dilate(masked_img, np.ones((3, 3)))
    return masked_img


def apply_smoothing(values, window_size=15):
    smoothed = []

    for i in range(1, len(values)):
        l = i - window_size // 2
        r = i + window_size // 2
        window = values[l: r]
        smoothed.append(np.mean(window))

    return smoothed


less_than_fn = lambda x, y: x < y
greater_than_fn = lambda x, y: x > y


def find_extrema_with_windowing(values, window_size=21, is_min=False):
    arg_fn = np.argmin if is_min else np.argmax
    comp_fn = less_than_fn if is_min else greater_than_fn

    extrema_points = []
    # for idx in range(window_size//2, len(values)-window_size//2):
    for idx in range(len(values)):
        l = idx - window_size // 2
        r = idx + window_size // 2
        window = values[l: r]
        if len(window) <= 1:
            continue
        local_extrema_idx = arg_fn(window)
        if idx == (l + local_extrema_idx):
            avg_before = np.mean(window[:len(window) // 2])
            avg_after = np.mean(window[len(window) // 2:])
            if comp_fn(values[idx], avg_before) and comp_fn(values[idx],
                                                            avg_after):
                extrema_points.append(idx)

    return extrema_points


def biggest_component(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8)

    mask = np.zeros(output.shape)
    if len(stats) == 1:
        return mask

    # Find the largest non background component.
    # Note: range() starts from 1 since 0 is the background label.
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(
        1, nb_components)], key=lambda x: x[1])

    mask[output == max_label] = 255
    return mask


def apply_threshold(maxs, masked_brain, img, thresh_val=150):
    # find the first maximum after the 150 and assume it is the hemorrhage
    hemorrhage_idx = 0
    for i, val in enumerate(maxs):
        if val > thresh_val:
            hemorrhage_idx = i
            break

    threshold = int(0.5 * maxs[hemorrhage_idx] + 0.5 * maxs[hemorrhage_idx - 1])

    width = maxs[-1] - threshold
    left_limit = maxs[-1] - width
    right_limit = maxs[-1] + width
    # print(left_limit, maxs[-1], right_limit)

    mask = np.copy(masked_brain)
    mask[mask < left_limit] = 0
    mask[mask > right_limit] = 0
    mask[mask > 0] = 255

    mask = erode_dilate_img(img, mask)

    segmented = np.copy(img)
    segmented[mask == 0] = 0

    # draw connected components and their areas
    output = cv2.connectedComponentsWithStats(mask)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    components_view = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    for label_idx in range(1, num_labels):  # starts from 1 to ignore backgronud
        bb_left = stats[label_idx, cv2.CC_STAT_LEFT]
        bb_top = stats[label_idx, cv2.CC_STAT_TOP]
        bb_height = stats[label_idx, cv2.CC_STAT_HEIGHT]
        bb_width = stats[label_idx, cv2.CC_STAT_WIDTH]
        bb_area = stats[label_idx, cv2.CC_STAT_AREA]

        cv2.rectangle(components_view, (bb_left, bb_top),
                      (bb_left + bb_width, bb_top + bb_height), (255, 0, 0), 1)
        cv2.putText(components_view, f"{bb_area}", (bb_left, bb_top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    return components_view, labels, stats, segmented, num_labels, mask


# assume that small connected components are noise and discard them
def mask_out_small_components(labels, stats, num_labels, area_threshold=20):
    components_to_keep = []
    for label_idx in range(1, num_labels):  # starts from 1 to ignore backgronud
        if stats[label_idx, cv2.CC_STAT_AREA] >= area_threshold:
            components_to_keep.append(label_idx)

    mask = np.zeros_like(labels)
    for label_idx in components_to_keep:
        mask[labels == label_idx] = 255

    return mask


def segment_ct_scan(img, min_mask=30, max_mask=230, thresh_val=150):
    img = resize_img(img)
    histogram = generate_histogram(img)
    mask = apply_mask_hide_pixel(img, min_mask, max_mask)

    # Erode + dilate
    masked_img = erode_dilate_img(img, mask)
    segmented = heatmap = masked_img
    if np.all(masked_img == 0):
        return segmented, heatmap
    # Connected components
    brain_mask = biggest_component(masked_img)
    masked_brain = np.copy(masked_img)
    masked_brain[brain_mask == 0] = 0
    histogram_brain_mask = generate_histogram(img, brain_mask.astype(np.uint8))

    smoothed_histogram = apply_smoothing(histogram_brain_mask, window_size=9)
    maxs = find_extrema_with_windowing(smoothed_histogram, window_size=7,
                                       is_min=False)

    if len(maxs) == 0:
        return segmented, heatmap
    components_view, labels, stats, segmented, num_labels, mask = \
        apply_threshold(maxs,
                        masked_brain,
                        img, thresh_val)
    noise_removal_mask = mask_out_small_components(labels, stats, num_labels)
    mask[noise_removal_mask == 0] = 0
    segmented[noise_removal_mask == 0] = 0

    heatmap = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    heatmap[mask > 0] = (255, 0, 0)

    return segmented, heatmap
