from Patient import PatientDataBase

# mods: gaussian, rectangle, noise, noise_gauss
mods = 'noise_gauss'
DATA_PATH_IN = "../../data/input_data"
DATA_PATH_OUT = "../data/output_data/" + mods
ct_data_out = "../data/output_data/png_org"


if __name__ == '__main__':
    print('start initialization')
    data_base = PatientDataBase(DATA_PATH_IN)
    gen = data_base.patient_generator()

    widths_gauss = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
    widths_rectangle = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]

    filtering = None
    if mods == 'gaussian' or mods == 'noise_gauss':
        filtering = 'gaussian'
    elif mods == 'rectangle':
        filtering = 'rectangle'
    elif not mods == 'noise':
        raise ValueError('mods not supported')

    for pat in gen:
        print(pat.id)
        for w in widths_gauss:
            pat.convolve_with_filter(width=w, filter_type=filtering)
            pat.write_modified_as_dicom(data_path=DATA_PATH_OUT, action='g')
        gen.__next__()
