# This code loads the CT slices (grayscale images) of the brain-window for each subject in ct_scans folder then saves them to
# one folder (data\image).
# Their segmentation from the masks folder is saved to another folder (data\label).

import os
from pathlib import Path

import nibabel as nib
import pandas as pd
from PIL import Image
from nibabel.spatialimages import SpatialImage
import numpy as np

from metrics_img import *
from segment_images import *


def load_ct_mask(datasetDir, sub_n, window_specs):
    ct_dir_subj = Path(datasetDir, 'ct_scans', "{0:0=3d}.nii".format(sub_n))
    ct_scan_nifti: SpatialImage = nib.load(str(ct_dir_subj))
    ct_scan_shape = ct_scan_nifti.shape
    ct_scan = np.asanyarray(ct_scan_nifti.dataobj)
    ct_scan = window_ct(ct_scan, ct_scan_shape, window_specs[0],
                        window_specs[1])  #
    # Convert the CT scans using a brain window
    # Loading the masks
    masks_dir_subj = Path(datasetDir, 'masks', "{0:0=3d}.nii".format(sub_n))
    masks_nifti = nib.load(str(masks_dir_subj))
    masks = np.asanyarray(masks_nifti.dataobj)
    return ct_scan, masks


def window_ct(ct_scan, ct_scan_shape, w_level=40, w_width=120):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    num_slices = ct_scan_shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:, :, s]
        # Faz a conversão da escala de hounsfield para 256 tons de cinza
        slice_s = (slice_s - w_min) * (255 / (w_max - w_min))
        slice_s[slice_s < 0] = 0
        slice_s[slice_s > 255] = 255
        ct_scan[:, :, s] = slice_s

    return ct_scan


def read_ct_scans(min_mask=30, max_mask=230, thresh_val=150, window_specs=[40, 80]):
    numSubj = 82
    new_size = (512, 512)
    currentDir = Path(os.getcwd())
    datasetDir = str(Path(currentDir))

    data_path = Path('data')

    # Reading labels
    hemorrhage_diagnosis_df = pd.read_csv(
        Path(data_path, 'hemorrhage_segment_results.csv'))

    hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values
    '''columns=['PatientNumber','SliceNumber','Intraventricular','Intraparenchymal','Subarachnoid','Epidural',
                                                                              'Subdural', 'No_Hemorrhage']) '''


    # reading images
    str_params = 'th_' + '_'.join(map(str, [min_mask, max_mask, thresh_val]))\
                 + '_.w_' + \
                 '_'.join(
        map(
            str,
            window_specs))

    idx_No_hemmo = "No_Hemmorhage_" + str_params
    idx_Jaccard = "Jaccard_" + str_params
    idx_Dice = "Dice_" + str_params
    idx_Specificity = "Specificity_" + str_params
    idx_F1Score = "F1Score_" + str_params
    idx_Sensitivity = "Sensitivity_" + str_params
    idx_Precision = "Precision_" + str_params

    hemorrhage_diagnosis_df[idx_No_hemmo] = np.nan
    hemorrhage_diagnosis_df[idx_Jaccard] = np.nan
    hemorrhage_diagnosis_df[idx_Dice] = np.nan
    hemorrhage_diagnosis_df[idx_Specificity] = np.nan
    hemorrhage_diagnosis_df[idx_F1Score] = np.nan
    hemorrhage_diagnosis_df[idx_Sensitivity] = np.nan
    hemorrhage_diagnosis_df[idx_Precision] = np.nan

    index_df = 0


    str_params_path = data_path / str_params
    image_path = str_params_path / 'image'
    label_path = str_params_path / 'label'

    segment_path = str_params_path / 'segment'

    heatmap_path = str_params_path / 'heatmap'
    if not data_path.exists():
        data_path.mkdir()
    str_params_path.mkdir()
    image_path.mkdir()
    label_path.mkdir()
    segment_path.mkdir()
    heatmap_path.mkdir()

    for sNo in range(0 + 49, numSubj + 49):
        if sNo > 58 and sNo < 66:  # no raw data were available for these subjects
            index_df += 1
            next
        else:
            # Loading the CT scan and masks
            ct_scan, masks = load_ct_mask(datasetDir, sNo, window_specs)

            idx = hemorrhage_diagnosis_array[:, 0] == sNo
            sliceNos = hemorrhage_diagnosis_array[idx, 1]

            if sliceNos.size != ct_scan.shape[2]:
                print(
                    'Warning: the number of annotated slices does not equal the number of slices in NIFTI file!')

            # Grouping image/label for subject
            image_sNo_path = image_path / str(sNo)
            label_sNo_path = label_path / str(sNo)
            segment_sNo_path = segment_path / str(sNo)
            heatmap_sNo_path = heatmap_path / str(sNo)
            image_sNo_path.mkdir()
            label_sNo_path.mkdir()
            segment_sNo_path.mkdir()
            heatmap_sNo_path.mkdir()

            for sliceI in range(0, sliceNos.size):
                ct_scan_slice = ct_scan[:, :, sliceI]
                mask_slice = resize_img(masks[:, :, sliceI])

                # Saving the a given CT slice
                x = Image.fromarray(ct_scan_slice).resize(new_size)
                x.convert("L").save(image_sNo_path / (str(sliceI) + '.png'))

                # Saving the segmentation for a given slice
                segmented, heatmap = segment_ct_scan(ct_scan_slice, min_mask,
                                                     max_mask, thresh_val)

                x = Image.fromarray(segmented).resize(new_size)
                x.convert("L").save(
                    segment_sNo_path / (str(sliceI) + '_HGE_Seg.jpg'))

                x = Image.fromarray(heatmap).resize(new_size)
                x.convert("L").save(
                    heatmap_sNo_path / (str(sliceI) + '_Heatmap.jpg'))

                # Saving mask
                x = Image.fromarray(mask_slice).resize(new_size)
                x.convert("L").save(label_sNo_path / (str(sliceI) + '.png'))

                hemorrhage_diagnosis_df.loc[index_df, idx_No_hemmo] = 1 if \
                    np.all(segmented == 0) else 0
                hemorrhage_diagnosis_df.loc[
                    index_df, idx_Jaccard] = jaccard_similarity(
                    mask_slice,
                    segmented)
                hemorrhage_diagnosis_df.loc[
                    index_df, idx_Dice] = dice_similarity(
                    mask_slice, segmented)
                hemorrhage_diagnosis_df.loc[
                    index_df, idx_Specificity] = calculate_specificity(
                    mask_slice, segmented)
                hemorrhage_diagnosis_df.loc[
                    index_df, idx_F1Score] = calculate_f1_score(
                    mask_slice, segmented)
                hemorrhage_diagnosis_df.loc[
                    index_df, idx_Sensitivity] = calculate_sensitivity(
                    mask_slice, segmented)
                hemorrhage_diagnosis_df.loc[
                    index_df, idx_Precision] = calculate_precision(
                    mask_slice, segmented)

                index_df += 1

    result_path = data_path / 'hemorrhage_segment_results.csv'
    hemorrhage_diagnosis_df.to_csv(result_path, index=False)

