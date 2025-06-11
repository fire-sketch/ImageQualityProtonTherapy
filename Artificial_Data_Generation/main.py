from Patient import PatientDataBase

# Choose processing mode: 'spatial', 'noise', 'combined'
# spatial: does image filtering
# noise: just adds noise to image data
# combined: adds noise after filtering image data with gaussian kernel
MODE = 'combined'

# Paths
DATA_PATH_IN = "../data/input_data"
DATA_PATH_OUT = "../data/output_data"

# Filter settings
FILTER_TYPE = 'gaussian'  # 'gaussian' or 'rectangle'


def main():
    """Main processing function with selectable modes."""
    print(f'Starting {MODE} processing')

    data_base = PatientDataBase(DATA_PATH_IN)
    gen = data_base.patient_generator()

    widths_gauss = [4, 10]
    widths_rectangle = [8, 20]
    noise_levels = [7.51, 20.55]

    # Select widths based on filter type
    if FILTER_TYPE == 'gaussian':
        widths = widths_gauss
    else:
        widths = widths_rectangle

    output_path = f"{DATA_PATH_OUT}/{MODE}_{FILTER_TYPE}"

    # ===========================
    # SPATIAL FILTERING MODE
    # ===========================
    if MODE == 'spatial':
        for pat in gen:
            print(pat.id)
            for w in widths:
                pat.convolve_with_filter(width=w, filter_type=FILTER_TYPE)
                pat.write_modified_as_dicom(data_path=output_path, mods=FILTER_TYPE)

    # ===========================
    # COMBINED SPATIAL + NOISE
    # ===========================
    elif MODE == 'combined':
        # widths_subset = [2, 5, 9, 16]  # Reduced set for combined processing

        for pat in gen:
            print(pat.id)
            for w in widths:
                for n in noise_levels:
                    pat.convolve_with_filter(width=w, filter_type=FILTER_TYPE)
                    pat.add_noise(n)
                    pat.write_modified_as_dicom(data_path=output_path, mods='noise_gauss')

    # ===========================
    # NOISE ONLY MODE
    # ===========================
    elif MODE == 'noise':
        for pat in gen:
            print(pat.id)
            for noise_level in noise_levels:
                pat.add_noise(noise_level)
                pat.write_modified_as_dicom(data_path=output_path, mods='noise')

    else:
        raise ValueError(f'Mode {MODE} not supported')

    print('Processing completed')


if __name__ == '__main__':
    main()
