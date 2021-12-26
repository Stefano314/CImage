import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time as tm
from scipy.ndimage import convolve
from scipy.ndimage import generic_filter

def local_intensity_sum(integral, window, x, y):
    '''
    Description
    -----------
    Returns the sum of all the pixel intensities within the local window, using the image total integral to speed up
    the process.

    Parameters
    ----------
    integral : numpy.array
        Gives the total integral intensity sum of the whole image.
    window : int
        Gives the linear dimension of the local window.
    x : int
        X-position of the considered pixel.
    y : int
        Y-position of the considered pixel.

    Returns
    -------
    output : int
    '''
    d = int(np.round(window/2 + 0.1))
    return integral[x+d-1][y+d-1] + integral[x-d][y-d] - integral[x-d][y+d-1] - integral[x+d-1][y-d]

def local_mean_intensity(loc_sum, window):
    '''
    Description
    -----------
    Returns the mean intensity of the pixels within the local window.

    Parameters
    ----------
    loc_int_sum : float
        Gives the local intensity sum of a specific pixel within a window.
    window : int
        Gives the linear dimension of the local window.

    Returns
    -------
    output : float
    '''
    return loc_sum/(window**2)

def global_threshold(image, threshold, vectorized=True):
    '''
    Description
    -----------
    Perform a global binary thresholding on a generic numpy array. Values higher than the threshold will be set to 1,
    while the remaining ones to 0.

    Parameters
    ----------

    image : numpy.array
        Intensity matrix of an image.
    threshold : int
        Sets the value of the threshold.
    vectorized : bool, optional
        Sets the mode with processing the image, when 'False' it is plain python, with 'True' it is vectorized.
        
    Returns
    -------
    output : numpy.array
    '''
    work_image = np.copy(image).astype(np.int64)
    if vectorized == False:
        for i in range(image.shape[0]):  # y values, removing borders
            for j in range(image.shape[1]):  # x values, removing borders
                if image[i, j] >= threshold:
                    work_image[i, j] = 255
                else:
                    work_image[i, j] = 0
    else:
        work_image = np.copy(image)
    return np.where(work_image >= threshold, 255, 0)

def bernsen_threshold(image, window = 13, vectorized=True):
    '''
    Description
    -----------
    Perform a local binary thresholding on a generic numpy array. Values higher than the threshold will be set to 1,
    while the remaining ones to 0. The threshold is calculated according to Bernsen’s technique.

    Parameters
    ----------
    image : numpy.array
        Intensity matrix of an image.
    window : int, optional
        Gives the linear dimension of the local window.
        The default value is '31'.
    vectorized : bool, optional
        Sets the mode with processing the image, when 'False' it is plain python, with 'True' it is vectorized.
        
    Returns
    -------
    output : numpy.array
    '''
    work_image = np.copy(image).astype(np.int64)
    d = int(np.round(window / 2 + 0.1))
    if vectorized == False:
        work_image[:d, :] = 0  # Remove left column
        work_image[:, :d] = 0  # Remove top row
        work_image[image.shape[0] - d:, :] = 0  # Remove right column
        work_image[:, image.shape[1] - d:] = 0  # Remove bottom row
        for i in range(d, image.shape[0] - d):  # y values, removing borders
            for j in range(d, image.shape[1] - d):  # x values, removing borders
                if image[i, j] >= 0.5 * (np.max(image[i - d:i + d + 1, j - d:j + d + 1]) +
                                         np.min(image[i - d:i + d + 1, j - d:j + d + 1])):
                    work_image[i, j] = 255
                else:
                    work_image[i, j] = 0
    else:
        def method(A):
            return 0.5*(np.max(A)-np.min(A))
        threshold = generic_filter(image, method, size = window, mode='constant')
        work_image = np.where(image >= threshold, 255, 0)
    return work_image

def niblack_threshold(image, window = 13, k = -0.2, vectorized=True):
    '''
    Description
    -----------
    Perform a local binary thresholding on a generic numpy array. Values higher than the threshold will be set to 1,
    while the remaining ones to 0. The threshold is calculated according to Niblack’s technique.

    Parameters
    ----------

    image : numpy.array
        Intensity matrix of an image.
    window : int, optional
        Gives the linear dimension of the local window.
        The default value is '11'.
    k : float, optional
        Parameter that tunes the thresholding value.
        The default value is '-0.2'.
    vectorized : bool, optional
        Sets the mode with processing the image, when 'False' it is plain python, with 'True' it is vectorized.
        
    Returns
    -------
    output : numpy.array
    '''
    work_image = np.copy(image).astype(np.int64)
    if vectorized == False:
        d = int(np.round(window / 2 + 0.1))
        work_image = np.copy(image).astype(np.int64)
        image_integral = image.cumsum(axis=0).cumsum(axis=1)
        work_image[:d, :] = 0 # Remove left column
        work_image[:, :d] = 0 # Remove top row
        work_image[image.shape[0]-d:, :] = 0 # Remove right column
        work_image[:, image.shape[1]-d:] = 0 # Remove bottom row
        for i in range(d, image.shape[0]-d): # y values, removing borders
            for j in range(d, image.shape[1]-d): # x values, removing borders
                if image[i,j] >= local_intensity_sum(image_integral,window,i,j)/window**2 + k*np.std(image[i-d:i+d+1,j-d:j+d+1]):
                    work_image[i,j] = 255
                else: work_image[i,j] = 0
    else:
        ker = np.ones((window, window)) / window ** 2
        loc_mean = convolve(image.astype(np.float32), ker, mode = 'constant')
        a = convolve((image - loc_mean) ** 2, ker, mode = 'constant')  # E[(X-E[X])^2]
        loc_std = np.sqrt(a)
        threshold = loc_mean + k * loc_std
        work_image = np.where(work_image >= threshold, 255, 0)
    return work_image

def sauvola_threshold(image, window = 13, k = 0.1, vectorized=True):
    '''
    Description
    -----------
    Perform a local binary thresholding on a generic numpy array. Values higher than the threshold will be set to 1,
    while the remaining ones to 0. The threshold is calculated according to Sauvola’s technique.

    Parameters
    ----------
    image : numpy.array
        Intensity matrix of an image.
    window : int, optional
        Gives the linear dimension of the local window.
        The default value is '11'.
    k : float, optional
        Parameter that tunes the thresholding value.
        The default value is '0.4'.
    vectorized : bool, optional
        Sets the mode with processing the image, when 'False' it is plain python, with 'True' it is vectorized.

    Returns
    -------
    output : numpy.array
    '''
    work_image = np.copy(image).astype(np.int64)
    if vectorized == False:
        image_integral = image.cumsum(axis = 0).cumsum(axis = 1).astype(np.int64)
        d = int(np.round(window / 2 + 0.1))
        work_image[:d, :] = 0  # Remove left column
        work_image[:, :d] = 0  # Remove top row
        work_image[image.shape[0] - d:, :] = 0  # Remove right column
        work_image[:, image.shape[1] - d:] = 0  # Remove bottom row
        for i in range(d, image.shape[0] - d):  # y values, removing borders
            for j in range(d, image.shape[1] - d):  # x values, removing borders
                if image[i, j] >= local_mean_intensity(local_intensity_sum(image_integral, window, i, j), window) * \
                        (1 + k*(np.std(image[i - d:i + d + 1, j - d:j + d + 1])/128 - 1)):
                    work_image[i, j] = 255
                else:
                    work_image[i, j] = 0
    else:
        ker = np.ones((window,window))/window**2
        loc_mean = convolve(image.astype(np.float32), ker, mode = 'constant')
        a = convolve((image-loc_mean)**2, ker, mode = 'constant') # E[(X-E[X])^2]
        loc_std = np.sqrt(a)
        threshold = loc_mean * (1 + k*(loc_std/128. - 1))
        work_image = np.where(work_image >= threshold, 255, 0)
    return work_image

def singh_threshold(image, window = 13, k = 0.03, vectorized=True):
    '''
    Description
    -----------
    Perform a local binary thresholding on a generic numpy array. Values higher than the threshold will be set to 1,
    while the remaining ones to 0. The threshold is calculated according to the "New Proposed" technique.

    Parameters
    ----------
    image : numpy.array
        Intensity matrix of an image.
    window : int, optional
        Gives the linear dimension of the local window.
        The default value is '11'.
    k : float, optional
        Parameter that tunes the thresholding value.
        The default value is '1'.
    vectorized : bool, optional
        Sets the mode with processing the image, when 'False' it is plain python, with 'True' it is vectorized.
        
    Returns
    -------
    output : numpy.array
    '''
    work_image = np.copy(image).astype(np.int64)
    if vectorized == False:
        image_integral = image.cumsum(axis = 0).cumsum(axis = 1).astype(np.int64)
        d = int(np.round(window / 2 + 0.1))
        work_image[:d, :] = 0  # Remove left column
        work_image[:, :d] = 0  # Remove top row
        work_image[image.shape[0] - d:, :] = 0  # Remove right column
        work_image[:, image.shape[1] - d:] = 0  # Remove bottom row
        for i in range(d, image.shape[0] - d):  # y values, removing borders
            for j in range(d, image.shape[1] - d):  # x values, removing borders
                m = local_mean_intensity(local_intensity_sum(image_integral, window, i, j), window)
                if image[i, j] >= m * (1 + k*((image[i, j]-m)/(255-image[i, j] + m) - 1)):
                    work_image[i, j] = 255
                else:
                    work_image[i, j] = 0
    else:
        ker = np.ones((window,window))/window**2
        loc_mean = convolve(image.astype(np.float32), ker, mode = 'constant')
        threshold = loc_mean * (1 + k*( (image-loc_mean)/(255-image+loc_mean) -1) )
        work_image = np.where(work_image >= threshold, 255, 0)
    return work_image

func_list = [global_threshold, bernsen_threshold, niblack_threshold, sauvola_threshold, singh_threshold]
labels = ['Global Technique', 'Bersen Technique', 'Niblack Technique', 'Sauvola Technique', 'Singh Technique']
# Original image plot
image = Image.open('image_test1.png').convert("L")
image_np = np.asarray(image).astype(np.int16)
thres_value, index = np.round(np.mean(image_np)), 0
print('Resolution:', image_np.shape)
plt.imshow(image_np, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Original Image', size = 15, fontweight = 'bold')
plt.show()

# Thresholds Plots
for threshold in func_list:
    if threshold == global_threshold:
        beg = tm.perf_counter()
        t_image = threshold(image_np, thres_value, vectorized = True)
        print(labels[index]+':', np.around(tm.perf_counter() - beg, 4),'s')
    else:
        beg = tm.perf_counter()
        t_image = threshold(image_np, vectorized = True)
        print(labels[index]+':', np.around(tm.perf_counter() - beg, 2),'s')
    plt.imshow(t_image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(labels[index], size = 15, fontweight = 'bold')
    index += 1
    plt.show()
