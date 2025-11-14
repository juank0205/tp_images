from skimage.io import imread, imshow
from  skimage.feature import peak_local_max
import scipy
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

def harris(I, sigma, K, t):
    I = I.astype('double')
    Ix = scipy.ndimage.sobel(I, axis=0)
    Iy = scipy.ndimage.sobel(I, axis=1)

    M1 = np.multiply(Ix, Ix)
    M2 = np.multiply(Iy, Ix)
    M4 = np.multiply(Iy, Iy)

    M1 = scipy.ndimage.gaussian_filter(M1, sigma)
    M2 = scipy.ndimage.gaussian_filter(M2, sigma)
    M4 = scipy.ndimage.gaussian_filter(M4, sigma)

    C = (np.multiply(M1, M4) - np.multiply(M2, M2)) - K * np.multiply(M1+M4, M1+M4)

    # plt.imshow(C, "gray")
    # plt.show()

    C2 = C.copy()
    C[C<t] = 0
    local_maxima = peak_local_max(C, min_distance=2)
    # centers = local_maxima.astype(float)
    centers = local_maxima

    C2 = C2 - np.min(C2)
    C2 = C2 / np.max(C2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show each image
    axes[0].imshow(I, "gray")
    axes[0].plot(centers[:, 1], centers[:, 0], 'o')
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(exposure.equalize_adapthist(C2), "gray")
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# I = imread('checkerboard.python.png')
# harris(I, 1, 0.04, 0)

I = imread('harris_swedenroad.python.png')
harris(I, 3, 0.04, 10**7)
