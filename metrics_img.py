import numpy as np


def jaccard_similarity(image1, image2):
    # achatar as matrizes para uma matriz unidimensional
    image1 = image1.flatten()
    image2 = image2.flatten()

    # Calculando a interseção e união de elementos
    intersection = np.intersect1d(image1, image2)
    union = np.union1d(image1, image2)

    # Calculando o coeficiente de Jaccard
    jaccard_coef = len(intersection) / len(union)

    return jaccard_coef


def dice_similarity(image1, image2):
    # achatar as matrizes para uma matriz unidimensional
    image1 = image1.flatten()
    image2 = image2.flatten()

    # Calculando a interseção de elementos
    intersection = np.intersect1d(image1, image2)

    # Calculando o coeficiente de Dice
    dice_coef = 2 * len(intersection) / (len(image1) + len(image2))

    return dice_coef