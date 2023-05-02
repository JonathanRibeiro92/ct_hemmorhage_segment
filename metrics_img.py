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

    # calcular o número de pixels corretamente classificados
    true_positives = np.sum(np.logical_and(image1 == 1, image2 == 1))

    # calcular o número de pixels que deveriam ter sido classificados como positivos
    actual_positives = np.sum(image1 == 1)

    # verificar se não há pixels positivos na imagem de referência
    if actual_positives == 0:
        sensitivity = 0.0
    else:
        # calcular o recall
        sensitivity = true_positives / actual_positives

    return sensitivity


def calculate_precision(image1, image2):
    # verificar se as imagens têm as mesmas dimensões
    if image1.shape != image2.shape:
        raise ValueError("As imagens têm dimensões diferentes!")

    # calcular o número de pixels corretamente classificados como positivos
    true_positives = np.sum(np.logical_and(image1 == 1, image2 == 1))

    # calcular o número de pixels classificados como positivos
    predicted_positives = np.sum(image2 == 1)

    # verificar se não há pixels positivos na imagem de referência
    if predicted_positives == 0:
        precision = 0.0
    else:
        # calcular a precisão
        precision = true_positives / predicted_positives

    return precision


def calculate_f1_score(image1, image2):
    # calcular a precisão e o recall
    precision = calculate_precision(image1, image2)
    recall = calculate_sensitivity(image1, image2)

    # calcular o F1 Score
    if precision == 0 and recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    return f1_score

#taxa de verdadeiros negativos
def calculate_specificity(image1, image2):
    # verificar se as imagens têm as mesmas dimensões
    if image1.shape != image2.shape:
        raise ValueError("As imagens têm dimensões diferentes!")

    # calcular o número de pixels corretamente classificados como negativos
    true_negatives = np.sum(np.logical_and(image1 == 0, image2 == 0))

    # calcular o número de pixels negativos na imagem de referência
    actual_negatives = np.sum(image1 == 0)

    # verificar se não há pixels negativos na imagem de referência
    if actual_negatives == 0:
        specificity = 0.0
    else:
        # calcular a especificidade
        specificity = true_negatives / actual_negatives

    return specificity