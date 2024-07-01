import glob

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy
from numpy.fft import fft, fftfreq
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.linear_model import LinearRegression
import Lookup_Data


def gauss(x, *p):
    a1, a2, a3, a4, a5 = p
    return a2 ** 2 * np.exp(-0.5 * ((x - a1) / a3) ** 2)


def pre_init():
    Lookup_Data.init_experiment_data()


def init(root, name,operation):
    params = None
    cor_x = None
    cor_y = None
    cor = None
    if operation == 'ESF':
        params = Lookup_Data.get_experiment_data(root, name, operation)

        cor_x = params.cor_x
        cor_y = params.cor_y
    elif operation == 'LSF' or operation == 'CT':
        params = Lookup_Data.get_experiment_data_ct(root, name)
        cor = params.cor


    slices = params.slices
    path = params.path
    factor = params.factor

    files = sorted(glob.glob(path + '/*.dcm'), key=len)
    data = []

    for file in files:
        data.append(pydicom.read_file(file))

    slices = data[slices[0]:slices[1]]
    s = np.array([s.pixel_array for s in slices])



    if operation == 'ESF':
        return s, cor_x, cor_y, data, factor
    else:
        return s, cor, data, factor



def esf(root, name, operation):
    s, cor_y, cor_x, data, factor = init(root, name,operation)
    p0 = [0, 1., 1., 1., 1.]
    if 'FDG' in name or 'GA' in name:
        p0 = [3, 1., 1., 1., 1.]
    z_summed = np.zeros((cor_x[1] - cor_x[0], cor_y[1] - cor_y[0]))
    for i, sli in enumerate(s):
        sl = sli[cor_x[0]:cor_x[1], cor_y[0]:cor_y[1]]

        z_summed = z_summed - sl
    ns = ['FDG_20']
    if name in ns:
        plt.imshow(z_summed,cmap='gray')
        plt.show()
    z_summed = z_summed / s.shape[0]
    esf_ = - np.sum(z_summed, axis=1) / z_summed.shape[1]
    if name in ns:
        plt.plot(esf_,marker='o')
        print(', '.join(map(str, esf_)))
        plt.show()
    lsf = np.diff(esf_)
    if name in ns:
        plt.plot(lsf,marker='o')
        print(', '.join(map(str, lsf)))
        plt.show()
    noise_ind = [0, 1, 2, -3, -2, -1]
    x = np.arange(len(lsf), dtype='float')
    x = x * data[0].PixelSpacing[0]
    x = x-x[np.argmax(lsf)]
    lsf = linear_detrend(lsf, noise_ind)

    lsf = lsf - np.mean(lsf[noise_ind])
    if name in ns:
        plt.plot(x, lsf,marker='o',label='x, detrend and denoise')
        print(', '.join(map(str, x)),'####',', '.join(map(str, lsf)))
        plt.legend()
        plt.show()
    function = gauss
    x_pred = np.arange(x[0], x[-1], 0.01 * data[0].PixelSpacing[0])
    p_opt, p_cov = curve_fit(function, x, lsf, p0=p0)
   # (p_opt)
    y_pred = function(x_pred, *p_opt)
   # plt.plot(x,lsf)
    #plt.plot(x_pred,y_pred)
    #plt.show()
    y_max = np.max(y_pred)
    y_pred = y_pred / y_max
    lsf = lsf / y_max


    x_max = x_pred[np.argmax(y_pred)]
    x = x - x_max
    x_pred = x_pred - x_max
    if name in ns:
        plt.plot(x, lsf, marker='o', label='normalize')
        plt.plot(x_pred, y_pred)
        print('lsf',', '.join(map(str, x)), '####', ', '.join(map(str, lsf)))
        print('x_pred',', '.join(map(str, x_pred)), '####', ', '.join(map(str, y_pred)))
        plt.show()
    return x, lsf, x_pred, y_pred


def zero_intersection(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1  # y!=0  y=mx+c
    x = -c / m
    return x


def calc_area_left(x, y):
    area = 0
    for i in range(1, len(y)):
        area += 0.5 * (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        if i > 1:
            area += (x[i] - x[i - 1]) * y[i - 1]
    return area


def calc_area_right(x, y):
    area = 0
    for i in range(len(y) - 2, -1, -1):
        area += 0.5 * (y[i] - y[i + 1]) * (x[i + 1] - x[i])
        if i < len(y) - 2:
            area += (x[i + 1] - x[i]) * y[i + 1]
    return area


def linear_detrend(y, noise_ind):
    x = np.arange(0, len(y)).reshape((-1, 1))
    trend_y = y[noise_ind]
    trend_x = x[noise_ind]
    model = LinearRegression()
    model.fit(trend_x, trend_y)
    for i, val in enumerate(y):
        y[i] = y[i] - (model.coef_ * x[i] + model.intercept_)
    return y


def calc_missing_area(left_x, left_y, right_x, right_y, left, right):
    search_points = np.linspace(left, right, 10)
    search_points = search_points[1:-1]
    min_area = 100
    x_mid = 0
    for p in search_points:
        a_left_calc = calc_area_left(np.append(left_x, p), np.append(left_y, 1))
        a_right_calc = calc_area_right(np.append(p, right_x), np.append(1, right_y))
        if np.abs(a_left_calc - a_right_calc) < min_area:
            min_area = np.abs(a_left_calc - a_right_calc)
            x_mid = p
    return x_mid


def find_intersection_with_zero(y,argmax=None):
    size = len(y) // 2
    if argmax:
        size = argmax
    y_left = y[0:size]
    y_right = y[size:-1]
    first_index = 0
    second_index = 0
    for i in range(0, len(y_left) - 1):
        if y_left[i] < 0:
            first_index = i

    for j in range(len(y_right) - 1, 0, -1):
        if y_right[j] < 0:
            second_index = j + size
    if first_index >= second_index:
        first_index = 0
    if second_index <= first_index:
        second_index = size - 1

    return first_index, second_index


def find_center(y):
    highest = np.argmax(y)
    left = highest - 1
    right = highest + 1
    first_index, second_index = find_intersection_with_zero(y)

    zero_left = zero_intersection(first_index, y[first_index], first_index + 1, y[first_index + 1])
    zero_right = zero_intersection(second_index - 1, y[second_index - 1], second_index, y[second_index])

    left_x = np.append(np.array([zero_left]), np.arange(first_index + 1, highest))
    left_y = np.append(np.array(0), y[first_index + 1:highest])

    right_x = np.append(np.arange(highest + 1, second_index), np.array([zero_right]))
    right_y = np.append(y[highest + 1:second_index], np.array(0))

    return calc_missing_area(left_x, left_y, right_x, right_y, left, right)


def do_lsf_radial(root, name, operation):
    p0 = [0, 1., 1., 1., 1.]
    if 'FDG' in name or 'GA' in name:
        p0 = [3, 1., 1., 1., 1.]
    s, cor, data, factor = init(root, name, operation)
    roi = 10
    s = s - 180.0
    z_summed = np.zeros((2 * roi, 2 * roi))

    print(name)
    for i, sli in enumerate(s):
        sl = sli[cor[1] - roi:cor[1] + roi, cor[0] - roi:cor[0] + roi]
        z_summed = z_summed - sl
    z_summed = z_summed / s.shape[0]
    #plt.imshow(z_summed,cmap='gray')
   # plt.show()
    angles = np.arange(0, 180, 20)
    bins = 20
    rebin_x = np.linspace(-5, 5, bins)  # -15 bis 19
    re_x = np.zeros(bins)
    rebin_y = np.zeros(bins)
    bin_counter = np.zeros(bins)

    for angle in angles:
        img_rot = ndimage.rotate(z_summed, angle, reshape=False, mode='reflect')
        img_rot = -factor * img_rot

        noise_ind = [0, 1, 2, 3, -4, -3, -2, -1]
        line_int = np.sum(img_rot, axis=0) / img_rot.shape[0]
        x = np.arange(len(line_int), dtype='float')
        #plt.plot(x,line_int,marker='o',label=f'erste LSF {name}')
        #print(', '.join(map(str, x)),'ääää',', '.join(map(str, line_int)))
       # plt.legend()
       # plt.show()
        line_int = linear_detrend(line_int, noise_ind)
       # plt.plot(x,line_int,marker='o',label='linear detrend')
       # print(', '.join(map(str, x)),'ääää',', '.join(map(str, line_int)))
        #plt.legend()
      #  plt.show()
        line_int = line_int - np.mean(line_int[noise_ind])
        line_int = line_int / np.max(line_int)
        ##plt.plot(x,line_int,marker='o',label='Normierung und Rauschen')
        ##print(', '.join(map(str, x)),'ääää',', '.join(map(str, line_int)))
       # plt.legend()
        #plt.show()
        mid = find_center(line_int)


        x = x - mid
        argmax = np.argmax(line_int)
      #  plt.plot(x,line_int,marker='o')
        #print(', '.join(map(str, x)),'ääää', ', '.join(map(str, line_int)))
       # plt.show()
        x[argmax] = 0
        x = x * data[0].PixelSpacing[0]
       # plt.plot(x,line_int,marker='o')
       # print(', '.join(map(str, x)),'ääää',', '.join(map(str, line_int)))
        #plt.legend()
        ##plt.grid()
        plt.title(name)
        plt.plot(x,line_int,label=f'{angle}')
        print(angle)
        print(', '.join(map(str, x)))
        print(', '.join(map(str, line_int)))
       # print(', '.join(map(str, x)),'ääää',', '.join(map(str, line_int)))
       # plt.show()
        plt.legend()
       # #if angle == 0:
           # print(angle)
       # print(f'{angle}: {fw_hm_org(x,line_int,argmax+1)}')
        bin_places = np.digitize(x, rebin_x)

        for i, bin_place in enumerate(bin_places):
            if bin_place != bins:
                re_x[bin_place] += x[i]
                rebin_y[bin_place] += line_int[i]
                bin_counter[bin_place] += 1


    for i, count in enumerate(bin_counter):
        if count != 0:
            rebin_y[i] /= count
            re_x[i] /= count
    rebin_x = re_x
    plt.show()
    to_delete = []
    for i in range(len(rebin_y)):
        if np.abs(rebin_y[i]) < 0.0001:
            to_delete.append(i)

    rebin_y = np.delete(rebin_y, to_delete)
    rebin_x = np.delete(rebin_x, to_delete)

    noise_ind = [0, 1, 2, 3, -4, -3, -2, -1]
    rebin_y = rebin_y - np.mean(rebin_y[noise_ind])
    rebin_y = rebin_y / np.max(rebin_y)

    rebin_x = rebin_x - rebin_x[np.argmax(rebin_y)]

    function = gauss
    pos_max = np.argmax(rebin_y)
    x_pred = np.arange(-pos_max, len(rebin_y) - pos_max, 0.01) * data[0].PixelSpacing[0]
    p_opt, p_cov = curve_fit(function, rebin_x, rebin_y, p0=p0)
    y_pred = function(x_pred, *p_opt)
    y_pred = y_pred / np.max(y_pred)

    spacing = data[0].PixelSpacing[0]
    x_max = x_pred[np.argmax(y_pred)]
    x_pred = x_pred - x_max
   # plt.plot(rebin_x,rebin_y)
    #print(', '.join(map(str, rebin_x)),'ääää',', '.join(map(str, rebin_y)))
    #print(', '.join(map(str, x_pred)),'ääää',', '.join(map(str, y_pred)))
    #plt.plot(x_pred,y_pred)
   # plt.show()
    return x_pred, y_pred, rebin_x, rebin_y, spacing


def do_mtf_org(x, y, spacing):
    if len(x) % 2 != 0:
        y = y[:-1]
        x = x[:-1]
    line = np.zeros(100000)
    line[50000 - int(len(x) / 2):50000 + int(len(x) / 2)] = y

    xf1, y1 = do_mtf(line, spacing)

    return xf1 * 10, y1


def do_mtf(line, spacing):
    yf = fft(line)
    x_f = np.arange(len(line)) * spacing
    n = len(x_f)
    xf1 = fftfreq(n, spacing)[:n // 2]
    y = np.abs(yf[0:n // 2])
    y1 = y / y[0]
    return xf1, y1


def do_mtf_fit(x, y, spacing):
    spacing = spacing * 0.01
    line = np.zeros(10000)
    line[5000 - int(len(x) / 2):5000 + int(len(x) / 2)] = y
    xf1, y1 = do_mtf(line, spacing)

    return xf1[0:100] * 10, y1[0:100]


def fw_hm_org(x, y,argmax=None):
    yminus = y - 0.5
    first_index, second_index = find_intersection_with_zero(yminus,argmax)
    x1 = x[first_index]
    x2 = x[first_index + 1]
    x3 = x[second_index - 1]
    x4 = x[second_index]
    x_left = zero_intersection(x1, yminus[first_index], x2, yminus[first_index + 1])
    x_right = zero_intersection(x3, yminus[second_index - 1], x4, yminus[second_index])
    fw_hm = np.round(np.abs(x_left) + x_right, 3)
    #if fw_hm < 1.0:
        #fw_hm = fw_hm_org(x,y,argmax)
   #if fw_hm > 3.0:
       # fw_hm = fw_hm_org(x,y,argmax)
    return fw_hm


def fw_hm_fit(x, y):
    y1 = np.argmin(np.abs(y[:int(len(y) / 2)] - 0.5))
    y2 = np.argmin(np.abs(y[int(len(y) / 2):-1] - 0.5))
    fw_hm = np.round(np.abs(x[y1]) + x[y2 + int(len(y) / 2)], 3)
    return fw_hm


def mtf_val_fit(x, y):
    x_50 = np.round(x[np.argmin(np.abs(y - 0.5))], 3)
    x_10 = np.round(x[np.argmin(np.abs(y - 0.1))], 3)
    return x_50, x_10


def mtf_val_org(x, y):
    index_50 = 0
    index_10 = 0
    y_50 = y - 0.5
    y_10 = y - 0.1
    for i in range(len(y)):
        if y_50[i] < 0:
            index_50 = i
            break
    for i in range(len(y)):
        if y_10[i] < 0:
            index_10 = i
            break
    x1 = x[index_50]
    x2 = x[index_50 + 1]
    x_50 = np.round(zero_intersection(x1, y_50[index_50], x2, y_50[index_50 + 1]), 3)
    x1 = x[index_10]
    x2 = x[index_10 + 1]
    x_10 = np.round(zero_intersection(x1, y_10[index_10], x2, y_10[index_10 + 1]), 3)
    return x_50, x_10
def noise_all_image(root,name):
    params = Lookup_Data.get_noise_data(root, name)
    cor = params.cor
    roi = 50
    if name == 'ub_039' or name == 'b_039':
        roi = 70
    if 'FDG' in name or 'GA' in name:
        cor = [220,220]
        roi = 12
   # print('roi ' + str(roi))
    mean,std = noi(roi,params,cor)
    #slices = params.slices
    #files = sorted(glob.glob(params.path + '/*.dcm'), key=len)
    #data = []

    #for file in files:
    #   data.append(pydicom.read_file(file))

   # slices = data[slices[0]:slices[1]]

    #s = np.array([s.pixel_array for s in slices])
    #sl = s[:, cor[1] - roi:cor[1] + roi, cor[0] - roi:cor[0] + roi]

    #hist_data = sl[10:20,25:35,25:35].flatten()
    #print(sl.shape)
   # for s in sl:

       # h = s[45:55,45:55].flatten()

      #  mean = np.mean(h)
       # std = np.std(h)
       # print(f'mean: {mean}, std: {std}')
        #if name == 'ub_039' or name == 'b_039':
       #    # h=s[65:75,65:75].flatten()
       # h=(h-mean)/std
    #print(sl.shape)
    #h = np.random.choice(sl.flatten(), 4999, replace=False)
    #h= sl[10:20,25:35,25:35].flatten()
        #print(stats.shapiro(h))
       # print(stats.kstest(h, stats.norm.cdf))
        #print(stats.kstest(h, stats.norm.cdf))
    #n, bins, patches = plt.hist(h,bins=30,density=True)
    #plt.show()
    # h = s[45:55,45:55].flatten()
   # notgood = 0
   # for j in range(sl.shape[0]):
     #   for i in range(sl.shape[2]):
#
      #      h = sl[j,:,i]
        #    mean = np.mean(h)
       #     std = np.std(h)
       # print(f'mean: {mean}, std: {std}')
        #if name == 'ub_039' or name == 'b_039':
        #    h=sl[12,:,70].flatten()
          #  h = (h - mean) / std
    # print(sl.shape)
    # h = np.random.choice(sl.flatten(), 4999, replace=False)
    # h= sl[10:20,25:35,25:35].flatten()
    # print(stats.shapiro(h))
         #   gauss = np.random.normal(0, 1, len(h))
            #stat,p = stats.shapiro(h)
        #    stat,p = stats.kstest(h, gauss)
         #   if p<0.05:
               # print(f'{j},{i},p-value: {p}')
          #      notgood +=1
   # z = np.shape(sl)[0]
    #x= np.shape(sl)[2]
   # y = np.shape(sl)[1]
   # print(f'Failquote y {100*notgood/(z*x)}')

   #for j in range(sl.shape[0]):
       # for i in range(sl.shape[1]):

         #   h = sl[j, i, :]
          #  mean = np.mean(h)
         #   std = np.std(h)
            # print(f'mean: {mean}, std: {std}')
            # if name == 'ub_039' or name == 'b_039':
            #    h=sl[12,:,70].flatten()
           # h = (h - mean) / std
            # print(sl.shape)
            # h = np.random.choice(sl.flatten(), 4999, replace=False)
            # h= sl[10:20,25:35,25:35].flatten()
            # print(stats.shapiro(h))
           # gauss = np.random.normal(0, 1, len(h))
            # stat,p = stats.shapiro(h)
           # stat, p = stats.kstest(h, gauss)
           # if p < 0.05:
                #print(f'{j},{i},p-value: {p}')
             #   notgood += 1
   # print(f'Failquote x {100 * notgood / (z * y)}')
    #for j in range(sl.shape[1]):
        #for i in range(sl.shape[2]):

            #h = sl[:, j, i]
           # mean = np.mean(h)
            #std = np.std(h)
            # print(f'mean: {mean}, std: {std}')
            # if name == 'ub_039' or name == 'b_039':
            #    h=sl[12,:,70].flatten()
            #h = (h - mean) / std
            # print(sl.shape)
            # h = np.random.choice(sl.flatten(), 4999, replace=False)
            # h= sl[10:20,25:35,25:35].flatten()
            # print(stats.shapiro(h))
          #  gauss = np.random.normal(0, 1, len(h))
            # stat,p = stats.shapiro(h)
          #  stat, p = stats.kstest(h, gauss)
          #  if p < 0.05:
                #print(f'{j},{i},p-value: {p}')
           #     notgood += 1

  #  print(f'Failquote z {100 * notgood / (x * y)}')
    #for i in range(sl.shape[0]):

      #  h = sl[i, 45:65,45:65].flatten()
      #  mean = np.mean(h)
       # std = np.std(h)
        # print(f'mean: {mean}, std: {std}')
        # if name == 'ub_039' or name == 'b_039':
        #    h=sl[12,:,70].flatten()
      #  h = (h - mean) / std
        # print(sl.shape)
        # h = np.random.choice(sl.flatten(), 4999, replace=False)
        # h= sl[10:20,25:35,25:35].flatten()
        # print(stats.shapiro(h))
        #gauss = np.random.normal(0, 1, len(h))
        # stat,p = stats.shapiro(h)
        #print(stats.kstest(h,gauss))

    #h = sl[10:20, 45:65, 45:65].flatten()
    #mean = np.mean(h)
    #std = np.std(h)
    # print(f'mean: {mean}, std: {std}')
    # if name == 'ub_039' or name == 'b_039':
    #    h=sl[12,:,70].flatten()
   # h = (h - mean) / std
    # print(sl.shape)
    # h = np.random.choice(sl.flatten(), 4999, replace=False)
    # h= sl[10:20,25:35,25:35].flatten()
    # print(stats.shapiro(h))
   # gauss = np.random.normal(0, 1, len(h))
    # stat,p = stats.shapiro(h)
#    print(stats.kstest(h, gauss))

    # if p < 0.05:
   #     print(f'p-value: {p}')
       # notgood += 1

    #print('mean')
    #print(me#an)
   # print('std')
    #print(std)
   # print(std / mean)

    #    s = np.random.normal(mean, np.std(data), len(data))

 #x_axis = np.arange(bins[np.argmax(n > 0.0001)], bins[np.argwhere(n > 0.0001)[-1]], 0.01)

 #   gaussian = np.random.normal(mean, std, len(x_axis))
    #plt.plot(x_axis, gaussian, color='orange')
#b = []
 #for i in np.arange(len(bins)-1):
     #   b.append((bins[i]+bins[i+1])/2.0)
   # print(len(b))
   # print(len(n))
    #b = np.array(b)
   # n =
    #plt.plot(b, 1/ (1 * np.sqrt(2 * np.pi)) *np.exp(- (b - 0) ** 2 / (2 * 1** 2)),
       #     linewidth=2, color='r')
#
   # plt.show()
   # print(stats.shapiro(h))
   # print(stats.kstest(hist_data, stats.norm.cdf))
   # testdata = sl[20,:,:].flatten()
   # testdata = (testdata-mean)/std
   # print(stats.shapiro(hist_data[:4999]))
    #hist_data = hist_data / np.max(hist_data)

   # hist_data = (hist_data - np.min(hist_data)) / (np.max(hist_data) - np.min(hist_data))
    #gauss = np.random.normal(mean, std, len(hist_data))
   # pdf = stats.norm.pdf(x_axis, loc=mean, scale=std)

    #def fit_function(k, *p0):
      #  lamb,loc = p0
      #  '''poisson function, parameter lamb is the fit parameter'''
      #  return stats.poisson.pmf(k, lamb,loc)
   # p0 = [1.0,mean]
    # fit with curve_fit
    #parameters, cov_matrix = curve_fit(fit_function, b, n,p0=p0)
    #normed_data = (hist_data - mean) / std
    #print(stats.normaltest(normed_data))
   # print(f'parametrs {parameters}')
   # print(stats.kstest(n, 'poisson',args=(10,0)))
    #n, bins, patches = plt.hist(hist_data, bins=75, density=True,ls='dotted', lw=3)
    #plt.hist(hist_data, bins=75)
    #plt.hist(gauss, bins=75)
    ##plt.show()

    return mean,std
def noi(roi,params,cor):
    slices = params.slices
    files = sorted(glob.glob(params.path + '/*.dcm'), key=len)
    data = []

    for file in files:
        data.append(pydicom.read_file(file))


    slices = data[slices[0]:slices[1]]


    s = np.array([s.pixel_array for s in slices])
    sl = s[:, cor[1] - roi:cor[1] + roi, cor[0] - roi:cor[0] + roi]
    #plt.imshow(sl[0], cmap='gray')
    #plt.show()
    #plt.imshow(sl[-1], cmap='gray')
   # plt.show()
    mean = np.round(np.mean(sl),2)
    std = np.round(np.std(sl),2)




    return mean, std
import math
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom
def noise_center(root,name):
    params = Lookup_Data.get_noise_data(root, name)
    cor = params.cor
    roi = 5
   # print('roi ' + str(roi))
    return noi(roi, params, cor)

def noise_not_center(root,name):
    params = Lookup_Data.get_noise_data(root, name)
    cor = [250,190]
    roi = 5
    if name == 'ub_039' or name == 'b_039':
        cor = [250,65]
    #print('roi ' + str(roi))
    return noi(roi, params, cor)