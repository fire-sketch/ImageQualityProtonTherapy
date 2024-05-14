import re
from pathlib import Path

import PatientEvaluation

do_overlay = 1
# gaussian, rect, noise, both, gauss_noise
mods = 'gaussian'
overlay_widths = [4,7,16]
folder_selected = None
if mods == 'rect':
    folder_selected = Path("../../data/rect")
if mods == 'gaussian':
    folder_selected = Path(r"../../data/gauss")
if mods == 'noise':
    folder_selected = Path("../data/noise")
if mods == 'gauss_noise':
    folder_selected = Path("../data/gauss_noise")

paths = [f for f in folder_selected.iterdir() if f.is_dir()]
patient_names = [p.name for p in paths]
print(patient_names)
values = input("Enter indices of patient to evaluate comma seperated:\n")
values = values.split(',')
values = [x.zfill(2) for x in values]
p = []
pat = []
for path, patient in zip(paths, patient_names):
    if re.search(r'\d{2}', patient)[0] in values:
        p.append(path)
        pat.append(patient)
paths = p
patient_names = pat
for p, pat in zip(paths, patient_names):
    PatientEvaluation.progress(do_overlay, str(p), pat, mods,overlay_widths=overlay_widths,overlay_ground='original')
