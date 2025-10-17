import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
import skimage
import progressbar

def booleanModel(windowSize, gamma, radius):
    edge = 2 * np.max(radius) + 100
    extendedWindow = windowSize + 2*edge
    windowArea = extendedWindow[0] * extendedWindow[1]
    nbPoints = np.random.poisson(lam=gamma * windowArea)

    x = np.random.rand(nbPoints)*extendedWindow[0]
    y = np.random.rand(nbPoints)*extendedWindow[1] # entero? ahora lo cambio

    grainsRadiuses = radius[0] + np.random.rand(nbPoints)*(radius[1] - radius[0])
    
    Z = np.zeros((extendedWindow[0], extendedWindow[1])).astype('int')
    for r, xi, yi in zip(grainsRadiuses, x, y):
        ri, ci = draw.disk((xi, yi), radius=r, shape=Z.shape)
        Z[ri, ci] = 1
    return Z

def minkowskiFunctionals(X):
    area = np.sum(X>0)
    perimeter = skimage.measure.perimeter(X, neighborhood=4)
    eulerNb = skimage.measure.euler_number(X, connectivity=2)
    return area, perimeter, eulerNb

def realizations(Wsize, gamma, radius, n=100):
    """
    This function iterates the different realizations
    Wsize: window size
    gamma: value of gamma, see booleanModel
    radius: min and max values of the radii of the generated disks
    """
    W = np.zeros((n, 3))
    areaWsize = Wsize[0] * Wsize[1]
    bar = progressbar.ProgressBar()
    for i in bar(range(n)):
        Z = booleanModel(Wsize, gamma, radius)
        a, p, chi = minkowskiFunctionals(Z)
        W[i, :] = np.array([a, p/2, chi*np.pi]) / areaWsize # huh?

    return W

window = [500, 500]
gamma = 100/(window[0]*window[1])
radius = [10, 30]
Z = booleanModel(window, gamma, radius)
plt.imshow(Z)
plt.show()

