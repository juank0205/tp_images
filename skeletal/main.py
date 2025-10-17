from scipy import ndimage
import matplotlib.pyplot as plt
import skimage
import numpy as np

element = np.array([[1, 1, 1], [0, 1, 0], [-1, -1, -1]])
element2 = np.array([[0, 1, 0], [1, 1, -1], [0, -1, -1]])
image = skimage.io.imread("./mickey.bmp")

def hitormiss(X, T):
    T1 = (T==1)
    T2 = (T==-1)
    temp1 = ndimage.binary_erosion(X, T1)
    temp2 = ndimage.binary_erosion(np.logical_not(X), T2)
    R = np.logical_and(temp1, temp2)
    return R

def elementary_thinning(X, T):
    R = np.minimum(X, np.logical_not(hitormiss(X, T)))
    return R

def elementary_thickenning(X, T):
    R = X ^ hitormiss(X, T)
    return R

def thinning(X, TT):
    for T in TT:
        X = elementary_thinning(X, T)
    return X

def topological_skeleton(X, T1, T2):
    temp = T1
    elements = []
    for _ in range(4):
        temp = np.rot90(temp)
        elements.append(temp)
    temp = T2
    for _ in range(4):
        temp = np.rot90(temp)
        elements.append(temp)
    
    B = np.logical_not(np.copy(X))
    while not(np.all(X==B)):
        B = X
        X = thinning(X, elements)
    return B

# skeleton = topological_skeleton(image, element, element2)
I = hitormiss(image, element)

plt.imshow(I)
plt.show()

