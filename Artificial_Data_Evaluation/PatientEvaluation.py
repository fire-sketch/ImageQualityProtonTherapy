from pathlib import Path
import natsort
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob
import os
import re
import Dicts


def get_cts(patient, use='mod', w='1', mods='gaussian'):
    path = ''
    if use == 'original':
        path = r"../../data/output_data/png_original" + "/" + patient
    elif use == 'mod':
        path = r"../../data/output_data/png/" + mods + '/' + patient + 'w' + w
    ct_paths = glob.glob(path + '/*.png')
    ct_paths = natsort.natsorted(ct_paths, reverse=True)
    cts = []
    for i, ct in enumerate(ct_paths):
        cts.append(np.asarray(PIL.Image.open(ct)))
    cts = np.dstack(cts)
    return cts


def draw(roi_img, contours, out_path, out_img, i):
    cv2.drawContours(roi_img, contours, -1, (0, 255, 0), 1)
    out_path = out_path + '/' + str(i) + '.png'
    scale_percent = 500
    width = int(roi_img.shape[1] * scale_percent / 100)
    height = int(roi_img.shape[0] * scale_percent / 100)
    size = (width, height)
    output = cv2.resize(roi_img, size)
    output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(out_path, output)
    out_img.append(output)


def make_overlay_gamma(patient, external, cts, roi, gammas, mods, name):
    data_path_out = "../../data/output_data/gamma_overlay/"
    data_path_out = data_path_out + mods + '/' + patient + '/' + name
    Path(data_path_out).mkdir(parents=True, exist_ok=True)
    roi = roi / 255
    roi = roi != 0.0000
    roi = roi.astype(np.uint8)
    ct_corner = np.asarray(Dicts.ct_corners[patient])
    grid_corner = np.asarray(Dicts.grid_corners[patient])
    corner_indexes = np.round(np.abs(ct_corner - grid_corner) * 10).astype(np.int)
    gammas[external == 0.0] = 0
    gammas = np.transpose(gammas, axes=[2, 1, 0])

    roi = np.transpose(roi, axes=[2, 1, 0])
    gammas = gammas.astype(cts.dtype)
    cts = cts[corner_indexes[1]:corner_indexes[1] + gammas.shape[1], corner_indexes[0]:corner_indexes[0]
                                                                                       + gammas.shape[0],
          corner_indexes[2]:corner_indexes[2] + gammas.shape[2]]
    #plt.imshow(cts[:, :, 50], cmap='gray')
    #plt.show()
    gammas = np.transpose(gammas, axes=[1, 0, 2])
    roi = np.transpose(roi, axes=[1, 0, 2])
    out_img = []
    for i in range(cts.shape[2]):
        # transversal
        out_path = data_path_out + '/transversal'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[:, :, i], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[:, :, i], cts[:, :, i], cts[:, :, i]))
        added_image = cv2.addWeighted(ct, 0.1, heatmap_img, 0.9, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[:, :, i])
        roi_img[gammas[:, :, i] == 0, ...] = ct[gammas[:, :, i] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[:, :, i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw(roi_img, contours, out_path, out_img, i)
    for i in range(cts.shape[1]):
        # sagittal
        out_path = data_path_out + '/sagittal'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[:, i, :], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[:, i, :], cts[:, i, :], cts[:, i, :]))
        added_image = cv2.addWeighted(ct, 0.1, heatmap_img, 0.9, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[:, i, :])
        roi_img[gammas[:, i, :] == 0, ...] = ct[gammas[:, i, :] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[:, i, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw(roi_img, contours, out_path, out_img, i)

    for i in range(cts.shape[0]):
        # coronal
        out_path = data_path_out + '/coronal'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[i, :, :], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[i, :, :], cts[i, :, :], cts[i, :, :]))
        added_image = cv2.addWeighted(ct, 0.1, heatmap_img, 0.9, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[i, :, :])
        roi_img[gammas[i, :, :] == 0, ...] = ct[gammas[i, :, :] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[i, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        draw(roi_img, contours, out_path, out_img, i)


def path_extraction(paths, folder_selected):
    path_evaluations = folder_selected + '/doses/*.npy'
    path_evaluations = glob.glob(path_evaluations)
    paths['path_evaluation'] = path_evaluations
    path_original = list(filter(lambda x: 'CT 1' in x, path_evaluations))[0]
    paths['path_original'] = path_original
    print('original file is ' + str(path_original))
    path_evaluations.remove(path_original)
    path_gamma = folder_selected + '/gamma/*.npy'
    path_gamma = glob.glob(path_gamma)
    paths['path_gamma'] = path_gamma
    print('gamma files are ' + str([os.path.splitext(os.path.basename(g))[0] for g in path_gamma]))
    paths['path_evaluations'] = path_evaluations
    print('evaluation files are ' + str([os.path.splitext(os.path.basename(p))[0] for p in path_evaluations]))
    path_roi = folder_selected + '/roi/*.npy'
    path_roi = glob.glob(path_roi)
    paths['path_roi'] = path_roi
    name_rois = [Path(filepath).stem for filepath in paths['path_roi']]
    paths['name_rois'] = name_rois


def evaluate_gamma(gamma, roi, mask, params, name):
    n = re.search(r'\d{1,2}\.?\d?', name).group()
    n = float(n)
    if 'gamma' in name:
        n = n * 2
    if 'gamma_noise' in name:
        noi = re.search(r'\d{1,3}\.\d{1,2}', name).group()
        noi = float(noi)
        params['noise'].append(noi)
    else:
        params['noise'].append(0)
    params['name'].append(n)
    roi_bin = roi != 0
    mask_2 = mask & roi_bin
    t2 = 1.5
    gam_trash = gamma[mask_2]
    valid_gamma = gam_trash[~np.isnan(gam_trash)]
    pass_rate = np.round(np.sum(valid_gamma <= 1) / len(valid_gamma) * 100, 2)
    mean_gamma = np.round(np.mean(valid_gamma), 2)  # modified median mean
    t2_gamma = np.sum(valid_gamma > t2) / len(valid_gamma)
    params['valid_gammas'].append(len(valid_gamma))
    params['pass_rate'].append(pass_rate)
    params['mean_gamma'].append(mean_gamma)
    params['T2_gamma'].append(np.round(t2_gamma, 2))


def clear_dic(dic):
    for key in dic:
        dic[key]: []


def progress(do_overlay, folder_selected, pat, mods,overlay_widths,overlay_ground='mod'):
    clear_dic(Dicts.paths)
    clear_dic(Dicts.dic_gamma)
    output_path = folder_selected + '/analysis_doses'
    os.makedirs(output_path, exist_ok=True)
    print('Selected folder ' + str(folder_selected))
    path_extraction(Dicts.paths, folder_selected)

    dose_original = np.load(Dicts.paths['path_original']) / 100
    doses_modified = [np.load(evaluation) / 100 for evaluation in Dicts.paths['path_evaluations']]
    gammas = [np.load(g) for g in Dicts.paths['path_gamma']]
    rois = [np.load(r) for r in Dicts.paths['path_roi']]

    max_dose = np.max(dose_original)
    dose_cutoff = 0.1 * max_dose
    relevant_slices = (np.max(dose_original, axis=(1, 2)) >= dose_cutoff)
    print('relevant ' + str(relevant_slices.sum()))
    print(f'sum {np.sum(dose_original)}')
    external = rois[0]
    rois_2 = []
    names = []
    for roi, name in zip(rois, Dicts.paths['name_rois']):
        if name == 'External_2':
            external = roi
            print('External')
            print(external.shape)
        else:
            print(name)
            print(roi.shape)
            rois_2.append(roi)
            names.append(name)

    for i, (gamma, dose_modified) in enumerate(zip(gammas, doses_modified)):
        print('gamma shape' + str(gamma.shape))
        regex = r'w(\d|\d\d).*.npy'
        name = re.search(regex, Dicts.paths['path_gamma'][i]).group()[:-4]
        w = re.search(r'\d\d|\d', name).group()
        print(f'Evaluating {name} gamma')
        if do_overlay and int(w) in overlay_widths:
            cts = get_cts(pat, overlay_ground, w, mods)
            show_gam = gamma.copy()
            show_gam[show_gam <= 1] = 0
            show_gam[show_gam > 1] = 255
            make_overlay_gamma(pat, external, cts, rois_2[0], show_gam, mods, name)
        maxi = Dicts.pres_dose[pat]
        mask = dose_modified > 0.1 * maxi
        evaluate_gamma(gamma, external, mask, Dicts.dic_gamma, name)
    df = pd.DataFrame.from_dict(Dicts.dic_gamma)
    df = df.sort_values(by=['name', 'noise'])
    patient = re.search(r'zzzCFPatient\d\d', folder_selected).group()
    df.to_csv(folder_selected + '/' + patient + '_analysis.csv')

    for i, roi in enumerate(rois_2):
        for key in Dicts.dic_roi:
            Dicts.dic_gamma[key]: []
        for j, (gamma, dose_modified) in enumerate(zip(gammas, doses_modified)):
            regex = r'w(\d|\d\d).*.npy'
            name = re.search(regex, Dicts.paths['path_gamma'][j]).group()[:-4]
            print(f'Evaluating {name} gamma roi')
            maxi = Dicts.pres_dose[pat]
            mask = dose_modified > 0.1 * maxi
            evaluate_gamma(gamma, roi, mask, Dicts.dic_roi, name)
        df = pd.DataFrame.from_dict(Dicts.dic_roi)
        df = df.sort_values(by=['name', 'noise'])
        patient = re.search(r'zzzCFPatient\d\d', folder_selected).group()
        df.to_csv(folder_selected + '/' + patient + '_roi' + names[i] + '_analysis.csv')
