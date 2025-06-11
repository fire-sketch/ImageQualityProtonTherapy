pres_dose = {
    'zzzCFPatient00': 50.4,
    'zzzCFPatient01': 54,
    'zzzCFPatient02': 54,
    'zzzCFPatient03': 14.4,
    'zzzCFPatient04': 54,
    'zzzCFPatient05': 21.6,
    'zzzCFPatient06': 69.96,
    'zzzCFPatient07': 23.4,
    'zzzCFPatient08': 45,
    'zzzCFPatient09': 54,
    'zzzCFPatient10': 54,
    'zzzCFPatient11': 50.4,
    'zzzCFPatient12': 54
}

dic_gamma = {
    'name': [],
    'noise': [],
    'valid_gammas': [],

    'pass_rate': [],
    'mean_gamma': [],
    'T2_gamma': [],
}

dic_roi = {
    'name': [],
    'noise': [],
    'valid_gammas': [],

    'pass_rate': [],
    'mean_gamma': [],
    'T2_gamma': [],
}

ct_corners = {
    'zzzCFPatient00': [-29.94, 16.21, 53.03], #links oben erste zahl,zweite zahl,höchste stelle dritte zahl
    'zzzCFPatient01': [-17.54, 12.61, 49.23],
    'zzzCFPatient02': [-17.53, 12.64, 52.27],
    'zzzCFPatient11': [-25.05, 18.15, 51.83]
}
grid_corners = {
    'zzzCFPatient00': [-4.36, -0.89, 54.73], #links oben erste zahl, zweite zahl,höchste stelle dritte zahl
    'zzzCFPatient01': [-4.7, 1.86, 59.93],
    'zzzCFPatient02': [-13.58, -1.34, 57.87],
    'zzzCFPatient11': [-12.8, 2.61, 58.33]
}

paths = {'path_reference': '',
         'path_original': '',
         'path_evaluations': '',
         'path_gamma': '',
         'path_roi': ''}

filtering = {
    '00': 'B',
    '01': 'UB',
    '02': 'UB',
    '03': 'B',
    '04': 'UB',
    '05': 'B',
    '06': 'B',
    '07': 'UB',
    '08': 'B',
    '09': 'UB',
    '10': 'UB',
    '11': 'UB',
    '12': 'UB',
    '13': 'UB'
}

max_energy = {
    '00': '152.5',
    '01': '147.2',
    '02': '158.7',
    '03': '190.9',
    '04': '155.6',
    '05': '187.7',
    '06': '203.2',
    '07': '172.2',
    '08': '148.5',
    '09': '122.6',
    '10': '159.7',
    '11': '178.2',
    '12': '168.9',
    '13': '168.9'
}

min_energy = {
    '00': '100',
    '01': '100',
    '02': '109.7',
    '03': '100',
    '04': '105.1',
    '05': '103.9',
    '06': '100',
    '07': '100',
    '08': '101.8',
    '09': '102.9',
    '10': '108',
    '11': '109.3',
    '12': '109.7',
    '13': '109.7'
}
lab = {
    '00': 'Thorax',
    '01': 'Gehirn',
    '02': 'Gehirn',
    '03': 'Abdomen',
    '04': 'Gehirn',
    '05': 'Abdomen',
    '06': 'HNO',
    '07': 'HNO',
    '08': 'Becken',
    '09': 'Auge/Gehirn',
    '10': 'Gehirn',
    '11': 'Gehirn',
    '12': 'Gehirn',
    '13': 'Becken'
}

