import numpy as np


def rates(true_image, pred_image):
    tp = np.sum(np.logical_and(true_image, pred_image))
    fp = np.sum(np.logical_and(pred_image, np.logical_not(true_image)))
    fn = np.sum(np.logical_and(true_image, np.logical_not(pred_image)))
    tn = np.sum(np.logical_and(np.logical_not(true_image), np.logical_not(pred_image)))
    return tp, fp, tn, fn

def jaccard_similarity(true_image, pred_image):
    tp, fp, _, fn = rates(true_image, pred_image)

    if (tp + fp + fn) == 0:
        return 0

    return tp / (tp + fp + fn)


def dice_similarity(true_image, pred_image):
    tp, fp, _, fn = rates(true_image, pred_image)
    if (tp + fp + fn) == 0:
        return 0

    return 2 * tp / (2 * tp + fp + fn)


# taxa de verdadeiros positivos ou taxa de recall
def calculate_sensitivity(true_image, pred_image):
    tp, _, _, fn = rates(true_image, pred_image)
    if (tp + fn) == 0:
        return 0

    return tp / (tp + fn)


def calculate_precision(true_image, pred_image):
    tp, fp, _, _ = rates(true_image, pred_image)
    if (tp + fp) == 0:
        return 0

    return tp / (tp + fp)


def calculate_f1_score(true_image, pred_image):
    # calcular a precisão e o recall
    precision = calculate_precision(true_image, pred_image)
    sensitivity = calculate_sensitivity(true_image, pred_image)

    if (precision + sensitivity) == 0:
        return 0

    return 2 * ((precision * sensitivity) / (precision + sensitivity))


# taxa de verdadeiros negativos
def calculate_specificity(true_image, pred_image):
    _, fp, tn, _ = rates(true_image, pred_image)
    if (tn + fp) == 0:
        return 0

    return tn / (tn + fp)


def calculate_false_positive_rate(true_image, pred_image):
    _, fp, tn, _ = rates(true_image, pred_image)
    if (tn + fp) == 0:
        return 0

    return fp/(fp + tn)

