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
    widths = [2, 5, 9, 16]
    noise = [7.51, 9.19, 10.28, 11.87, 14.53, 20.55]

    filtering = None
    if mods == 'gaussian' or mods == 'noise_gauss':
        filtering = 'gaussian'
    elif mods == 'rectangle':
        filtering = 'rectangle'
    elif not mods == 'noise':
        raise ValueError('mods not supported')

    for pat in gen:
        print(pat.id)
        for w in widths:
            for n in noise:
                pat.convolve_with_filter(width=w, filter_type=filtering)
                pat.add_noise(n)
                pat.write_modified_as_dicom(data_path=DATA_PATH_OUT, action=mods)
