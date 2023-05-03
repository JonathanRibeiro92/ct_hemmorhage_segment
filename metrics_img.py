import numpy as np


def jaccard_similarity(image1, image2):
    # Contar o número de pixels em comum entre as duas imagens
    intersection = np.logical_and(image1, image2).sum()

    # Contar o número de pixels na união das duas imagens
    union = np.logical_or(image1, image2).sum()

    if union == 0:
        # Se a união for zero, não é possível calcular o índice de Jaccard
        # Retorne NaN ou outro valor apropriado para a sua aplicação
        similarity = float('nan')
    else:
        # Calcular o índice de Jaccard
        similarity = intersection / union

    return similarity


def dice_similarity(image1, image2):
    # Contar o número de pixels em comum entre as duas imagens
    intersection = np.logical_and(image1, image2).sum()

    # Contar o número de pixels em cada imagem
    total_pixels = image1.sum() + image2.sum()

    if total_pixels == 0:
        # Se o número total de pixels for zero, não é possível calcular o coeficiente Dice
        # Retorne NaN ou outro valor apropriado para a sua aplicação
        similarity = float('nan')
    else:
        # Calcular o coeficiente Dice
        similarity = 2.0 * intersection / total_pixels

    return similarity


# taxa de verdadeiros positivos ou taxa de recall
def calculate_sensitivity(image1, image2):
    # verificar se as imagens têm as mesmas dimensões
    # if image1.shape != image2.shape:
    #     raise ValueError("As imagens têm dimensões diferentes!")

    # compara as imagens pixel a pixel
    comp = np.equal(image1, image2)

    # conta os verdadeiros positivos e falsos negativos
    tp = np.sum(comp)
    fn = np.sum(comp & (image2 == 0))

    if fn == 0:
        sensitivity = 0.0
    else:
        # calcula a sensibilidade
        sensitivity = tp / (tp + fn)

    return sensitivity


def calculate_precision(image1, image2):
    # verificar se as imagens têm as mesmas dimensões
    if image1.shape != image2.shape:
        raise ValueError("As imagens têm dimensões diferentes!")

    # compara as imagens pixel a pixel
    comp = np.equal(image1, image2)

    # conta os verdadeiros positivos e falsos positivos
    tp = np.sum(comp)
    fp = np.sum(~comp & (image2 == 1))

    # verificar se não há pixels positivos na imagem de referência
    if fp == 0:
        precision = 0.0
    else:
        # calcula a precisão
        precision = tp / (tp + fp)

    return precision


def calculate_f1_score(image1, image2):
    # calcular a precisão e o recall
    precision = calculate_precision(image1, image2)
    sensitivity = calculate_sensitivity(image1, image2)

    # calcular o F1 Score
    if precision == 0 and sensitivity == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))

    return f1_score


# taxa de verdadeiros negativos
def calculate_specificity(image1, image2):
    # verificar se as imagens têm as mesmas dimensões
    if image1.shape != image2.shape:
        raise ValueError("As imagens têm dimensões diferentes!")


    comp = (image1 != image2)

    # conta os verdadeiros negativos e falsos positivos
    tn = np.sum(comp & (image1 == 0) & (image2 == 0))
    fp = np.sum(comp & (image1 == 0) & (image2 == 1))

    if fp == 0:
        specificity = 0.0
    else:
        # calcula a especificidade
        specificity = tn / (tn + fp)

    return specificity
