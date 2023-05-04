import numpy as np


def jaccard_similarity(image1, image2):
    tp = np.sum(np.logical_and(image1, image2))
    fp = np.sum(np.logical_and(image1, np.logical_not(image2)))
    fn = np.sum(np.logical_and(np.logical_not(image1), image2))

    if (tp + fp + fn) < 1:
        return 0

    return tp / (tp + fp + fn)


def dice_similarity(image1, image2):
    tp = np.sum(np.logical_and(image1, image2))
    fp = np.sum(np.logical_and(image1, np.logical_not(image2)))
    fn = np.sum(np.logical_and(np.logical_not(image1), image2))

    if (tp + fp + fn) < 1:
        return 0

    return 2 * tp / (2 * tp + fp + fn)


# taxa de verdadeiros positivos ou taxa de recall
def calculate_sensitivity(image1, image2):
    tp = np.sum(np.logical_and(image1, image2))
    fn = np.sum(np.logical_and(np.logical_not(image1), image2))

    if (tp + fn) < 1:
        return 0

    return tp / (tp + fn)


def calculate_precision(image1, image2):
    tp = np.sum(np.logical_and(image1, image2))
    fp = np.sum(np.logical_and(image1, np.logical_not(image2)))

    if (tp + fp) < 1:
        return 0

    return tp / (tp + fp)


def calculate_f1_score(image1, image2):
    # calcular a precisão e o recall
    precision = calculate_precision(image1, image2)
    sensitivity = calculate_sensitivity(image1, image2)

    if (precision + sensitivity) < 1:
        return 0

    return 2 * ((precision * sensitivity) / (precision + sensitivity))


# taxa de verdadeiros negativos
def calculate_specificity(image1, image2):
    tn = np.sum(np.logical_and(np.logical_not(image1), np.logical_not(image2)))
    fp = np.sum(np.logical_and(image1, np.logical_not(image2)))

    if (tn + fp) < 1:
        return 0

    return tn / (tn + fp)
