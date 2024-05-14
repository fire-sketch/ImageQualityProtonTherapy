from Artificial_Data_Generation.Patient import PatientDataBase

# mods: gaussian, rectangle, noise, noise_gauss
mods = 'rectangle'
DATA_PATH_IN = "../../data/input_data"
DATA_PATH_OUT = "../../data/output_data/" + mods
ct_data_out = "../../data/output_data/png_org"


if __name__ == '__main__':
    print('start initialization')
    data_base = PatientDataBase(DATA_PATH_IN)
    gen = data_base.patient_generator()

  #  widths_gauss = [2,5,9,16]
    widths_rect = [50]
    widths_rectangle = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]

    filtering = None
    if mods == 'gaussian' or mods == 'noise_gauss':
        filtering = 'gaussian'
    elif mods == 'rectangle':
        filtering = 'rectangle'
    elif not mods == 'noise':
        raise ValueError('mods not supported')
    patients = ['zzzCFPatient11']
    for pat in gen:
        if pat in patients:
            print(pat.id)
            for w in widths_rect:
                pat.convolve_with_filter(w,filter_type=mods)
                pat.write_modified_as_png(mods=mods, safe_original=False, center=-200, width=1000)
            gen.__next__()
