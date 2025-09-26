from scipy import ndimage
import matplotlib.pyplot as plt
import skimage
import numpy as np

image = skimage.io.imread("./poudre.bmp")
image = image > 74
image = ndimage.binary_fill_holes(image)

se = ndimage.generate_binary_structure(2, 1)
m = ndimage.binary_opening(image)
image = ndimage.binary_propagation(m, mask=image)

def granulometry(X, n):
    total_area = ndimage.sum(X)
    label, total_objects = ndimage.label(X)

    area = np.zeros((35,), dtype=float)
    number = np.zeros((35,), dtype=float)

    structure = ndimage.generate_binary_structure(2, 1)
    for i in np.arange(n):
        S = ndimage.iterate_structure(structure, i-1)
        E = ndimage.binary_erosion(X, S)
        P = ndimage.binary_propagation(E, mask=X)

        area[i] = (100*ndimage.sum(P))/total_area
        label, count = ndimage.label(P)
        number[i] = 100*count/total_objects
        
    plt.figure()
    plt.plot(area, label="Area")
    plt.plot(number, label="Number")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(-np.diff(area), label="Area derivative")
    plt.plot(-np.diff(number), label="Number derivative")
    plt.legend()
    plt.show()

granulometry(image, 35)
