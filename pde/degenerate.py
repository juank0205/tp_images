from typing import Tuple
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

def degenerateDiffusion(I, nb_iterations: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    ZDilation = I.copy()
    ZErosion = I.copy()
    for _ in range(nb_iterations):
        EgN = np.roll(ZErosion, 1, axis=0) - ZErosion
        EgE = np.roll(ZErosion, -1, axis=1) - ZErosion
        EgS = np.roll(ZErosion, -1, axis=0) - ZErosion
        EgW = np.roll(ZErosion, 1, axis=1) - ZErosion

        DgN = np.roll(ZDilation, 1, axis=0) - ZDilation
        DgE = np.roll(ZDilation, -1, axis=1) - ZDilation
        DgS = np.roll(ZDilation, -1, axis=0) - ZDilation
        DgW = np.roll(ZDilation, 1, axis=1) - ZDilation

        ZDilation = ZDilation + dt * np.sqrt(np.minimum(0, -DgN)**2 + np.maximum(0, -DgE)**2 + np.minimum(0, -DgW)**2 + np.maximum(0, -DgS)**2)
        ZErosion = ZErosion - dt * np.sqrt(np.maximum(0, -EgN)**2 + np.minimum(0, -EgE)**2 + np.maximum(0, -EgW)**2 + np.minimum(0, -EgS)**2)
    return ZDilation, ZErosion

I = skimage.io.imread("./images/cerveau.png")/255
dilation_1, erosion_1 = degenerateDiffusion(I, 10, 0.05)
dilation_2, erosion_2 = degenerateDiffusion(I, 50, 0.05)

fig, axes = plt.subplots(2, 3, figsize=(10, 4))

axes[0, 0].imshow(I, cmap='gray')
axes[0, 0].set_title("Original")

axes[0, 1].imshow(dilation_1, cmap='gray')
axes[0, 1].set_title("Dilation 0.05dt 10itr")

axes[0, 2].imshow(erosion_1, cmap='gray')
axes[0, 2].set_title("Erosion 0.05dt 10itr")

axes[1, 1].imshow(dilation_2, cmap='gray')
axes[1, 1].set_title("Dilation 0.05dt 50itr")

axes[1, 2].imshow(erosion_2, cmap='gray')
axes[1, 2].set_title("Erosion 0.05dt 50itr")

plt.tight_layout()
plt.show()


