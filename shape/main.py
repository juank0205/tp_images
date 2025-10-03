import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import skimage
import scipy

def crofton_perimeter(I):
    """ Computation of crofton perimeter
    """
    inter = []
    h = np.array([[1, -1]])
    for i in range(4):
        II = np.copy(I)
        I2 = skimage.transform.rotate(II, angle=45*i, order=0)
        I3 = scipy.ndimage.convolve(I2, h)

        inter.append(0.5*np.sum(np.abs(I3)))

    crofton = np.pi/4. * (inter[0]+inter[2] + (inter[1]+inter[3])/np.sqrt(2))
    return crofton

def feret_diameter(I):
    """ 
    Computation of the Feret diameter
    minimum: d (meso-diameter)
    maximum: D (exo-diameter)

    Input: I binary image
    """
    d = np.max(I.shape)
    D = 0

    for a in np.arange(0, 180, 30):
        I2 = skimage.transform.rotate(I, angle=a, order=0)
        F = np.max(I2, axis=0)
        measure = np.sum(F )

        if (measure < d):
            d = measure
        if (measure > D):
            D = measure
    return d, D

def inscribedRadius(I):
    """
    computes the radius of the inscribed circle
    """
    dm = scipy.ndimage.distance_transform_cdt(I > 100)
    radius = np.max(dm)
    return radius

def elongation(d, D):
    return d/D

def area(I):
    return np.sum(I>100)

def circularity(a, D):
    return (4*a)/(np.pi*D**2)

def thinness(r, D):
    return (2*r)/D

I = imageio.imread('images/apple-1.bmp')
crofton = crofton_perimeter(I)
feret = feret_diameter(I)
radius = inscribedRadius(I)
plt.imshow(I)
plt.show()
print(f"Crofton perimeter: {crofton}")
print(f"Feret diameter: {feret}")
print(f"Inscribed circle radius: {radius}")
