from split_raw_data import *

import os
from pathlib import Path

import time


def read_CSVResults_calc_metrics(min_mask, max_mask, thresh_val, window_specs):
    result_path = Path('result')

    hemorrhage_diagnosis_df = pd.read_csv(
        Path(result_path, 'hemorrhage_segment_results.csv'))

    str_params = 'th_' + '_'.join(map(str, [min_mask, max_mask, thresh_val])) \
                 + '_.w_' + \
                 '_'.join(
                     map(
                         str,
                         window_specs))

    idx_Jaccard = "Jaccard_" + str_params
    idx_Dice = "Dice_" + str_params
    idx_Specificity = "Specificity_" + str_params
    idx_F1Score = "F1Score_" + str_params
    idx_Sensitivity = "Sensitivity_" + str_params
    idx_Precision = "Precision_" + str_params
    idx_FPR = "FPR_" + str_params

    min_jaccard = hemorrhage_diagnosis_df[idx_Jaccard].min()
    min_Dice = hemorrhage_diagnosis_df[idx_Dice].min()
    min_Specificity = hemorrhage_diagnosis_df[idx_Specificity].min()
    min_F1Score = hemorrhage_diagnosis_df[idx_F1Score].min()
    min_Sensitivity = hemorrhage_diagnosis_df[idx_Sensitivity].min()
    min_Precision = hemorrhage_diagnosis_df[idx_Precision].min()
    min_FPR = hemorrhage_diagnosis_df[idx_FPR].min()

    max_jaccard = hemorrhage_diagnosis_df[idx_Jaccard].max()
    max_Dice = hemorrhage_diagnosis_df[idx_Dice].max()
    max_Specificity = hemorrhage_diagnosis_df[idx_Specificity].max()
    max_F1Score = hemorrhage_diagnosis_df[idx_F1Score].max()
    max_Sensitivity = hemorrhage_diagnosis_df[idx_Sensitivity].max()
    max_Precision = hemorrhage_diagnosis_df[idx_Precision].max()
    max_FPR = hemorrhage_diagnosis_df[idx_FPR].max()

    mean_jaccard = hemorrhage_diagnosis_df[idx_Jaccard].mean()
    mean_Dice = hemorrhage_diagnosis_df[idx_Dice].mean()
    mean_Specificity = hemorrhage_diagnosis_df[idx_Specificity].mean()
    mean_F1Score = hemorrhage_diagnosis_df[idx_F1Score].mean()
    mean_Sensitivity = hemorrhage_diagnosis_df[idx_Sensitivity].mean()
    mean_Precision = hemorrhage_diagnosis_df[idx_Precision].mean()
    mean_FPR = hemorrhage_diagnosis_df[idx_FPR].mean()

    std_jaccard = hemorrhage_diagnosis_df[idx_Jaccard].std()
    std_Dice = hemorrhage_diagnosis_df[idx_Dice].std()
    std_Specificity = hemorrhage_diagnosis_df[idx_Specificity].std()
    std_F1Score = hemorrhage_diagnosis_df[idx_F1Score].std()
    std_Sensitivity = hemorrhage_diagnosis_df[idx_Sensitivity].std()
    std_Precision = hemorrhage_diagnosis_df[idx_Precision].std()
    std_FPR = hemorrhage_diagnosis_df[idx_FPR].std()

    # Definir colunas e índices
    colunas = ['Jaccard', 'Dice', 'Specificity', 'F1Score', 'Sensitivity',
               'Precision', 'FPR']
    indices = ['Min', 'Max', 'STD', 'Mean']

    # Criar dataframe vazio com colunas e índices
    df_metrics = pd.DataFrame(columns=colunas, index=indices)

    df_metrics.loc['Min', 'Jaccard'] = min_jaccard
    df_metrics.loc['Min', 'Dice'] = min_Dice
    df_metrics.loc['Min', 'Specificity'] = min_Specificity
    df_metrics.loc['Min', 'F1Score'] = min_F1Score
    df_metrics.loc['Min', 'Sensitivity'] = min_Sensitivity
    df_metrics.loc['Min', 'Precision'] = min_Precision
    df_metrics.loc['Min', 'FPR'] = min_FPR

    df_metrics.loc['Max', 'Jaccard'] = max_jaccard
    df_metrics.loc['Max', 'Dice'] = max_Dice
    df_metrics.loc['Max', 'Specificity'] = max_Specificity
    df_metrics.loc['Max', 'F1Score'] = max_F1Score
    df_metrics.loc['Max', 'Sensitivity'] = max_Sensitivity
    df_metrics.loc['Max', 'Precision'] = max_Precision
    df_metrics.loc['Max', 'FPR'] = max_FPR

    df_metrics.loc['STD', 'Jaccard'] = std_jaccard
    df_metrics.loc['STD', 'Dice'] = std_Dice
    df_metrics.loc['STD', 'Specificity'] = std_Specificity
    df_metrics.loc['STD', 'F1Score'] = std_F1Score
    df_metrics.loc['STD', 'Sensitivity'] = std_Sensitivity
    df_metrics.loc['STD', 'Precision'] = std_Precision
    df_metrics.loc['STD', 'FPR'] = std_FPR

    df_metrics.loc['Mean', 'Jaccard'] = mean_jaccard
    df_metrics.loc['Mean', 'Dice'] = mean_Dice
    df_metrics.loc['Mean', 'Specificity'] = mean_Specificity
    df_metrics.loc['Mean', 'F1Score'] = mean_F1Score
    df_metrics.loc['Mean', 'Sensitivity'] = mean_Sensitivity
    df_metrics.loc['Mean', 'Precision'] = mean_Precision
    df_metrics.loc['Mean', 'FPR'] = mean_FPR

    metrics_path = result_path / (str_params + 'metrics_segment_results.csv')
    df_metrics.to_csv(metrics_path)


def main():
    currentDir = Path(os.getcwd())
    datasetDir = str(Path(currentDir))

    result_path = Path('result')
    if not result_path.exists():
        result_path.mkdir()

    hemorrhage_diagnosis_df = pd.read_csv(
        Path(datasetDir, 'hemorrhage_diagnosis_raw_ct.csv'))

    hemorrhage_diagnosis_df = hemorrhage_diagnosis_df.drop(
        hemorrhage_diagnosis_df[hemorrhage_diagnosis_df.SliceNumber < 6].index)

    result_path = result_path / 'hemorrhage_segment_results.csv'
    hemorrhage_diagnosis_df.to_csv(result_path, index=False)

    params = {
        1: [(30, 230, 150), [40, 80]],
        2: [(30, 230, 155), [40, 80]],
        3: [(30, 230, 160), [40, 80]],
        4: [(30, 230, 165), [40, 80]],
        5: [(30, 230, 170), [40, 80]],
        6: [(30, 230, 175), [40, 80]],
        7: [(30, 230, 150), [40, 120]],
        8: [(30, 230, 155), [40, 120]],
        9: [(30, 230, 160), [40, 120]],
        10: [(30, 230, 165), [40, 120]],
        11: [(30, 230, 170), [40, 120]],
        12: [(30, 230, 175), [40, 120]],
        13: [(30, 230, 180), [40, 120]],
        14: [(30, 230, 185), [40, 120]],
        15: [(30, 230, 190), [40, 120]],
        16: [(30, 230, 195), [40, 120]],
        17: [(30, 230, 200), [40, 120]],
        18: [(30, 200, 175), [40, 120]],
        19: [(30, 210, 175), [40, 120]],
        20: [(30, 220, 175), [40, 120]],
        21: [(10, 220, 175), [40, 120]],
        22: [(20, 220, 175), [40, 120]],
        23: [(40, 220, 175), [40, 120]],
        24: [(60, 220, 175), [40, 120]],
        25: [(80, 220, 175), [40, 120]],
        26: [(100, 220, 175), [40, 120]],

    }

    for param in params.values():
        min_mask = param[0][0]
        max_mask = param[0][1]
        thresh_val = param[0][2]
        window_specs = param[1]
        read_ct_scans(min_mask=min_mask, max_mask=max_mask,
                      thresh_val=thresh_val,
                      window_specs=window_specs)
        read_CSVResults_calc_metrics(min_mask, max_mask, thresh_val,
                                     window_specs)


if __name__ == '__main__':
    print('Process starting...')
    start_time = time.time()
    main()
    print('Process ending!')
    print("--- %s seconds ---" % (time.time() - start_time))
