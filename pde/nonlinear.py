import skimage.io
import numpy as np
import matplotlib.pyplot as plt

def c(I: np.ndarray, alpha: float) -> np.ndarray:
    return np.exp(-(I/alpha)**2)

def nonLinearDiffusion(I: np.ndarray, nb_iterations: int, dt: float, alpha: float) -> np.ndarray:
    Z = I.copy()
    for _ in range(nb_iterations):
        gN = np.roll(Z, 1, axis=0) - Z
        gE = np.roll(Z, -1, axis=1) - Z
        gS = np.roll(Z, -1, axis=0) - Z
        gW = np.roll(Z, 1, axis=1) - Z
        Z = Z + dt * (c(np.abs(gN), alpha) * gN + c(np.abs(gE), alpha) * gE + c(np.abs(gS), alpha) * gS + c(np.abs(gW), alpha) * gW)
    return Z


I = skimage.io.imread("./images/cerveau.png")/255
filtered_1 = nonLinearDiffusion(I, 10, 0.05, 0.1)
filtered_2 = nonLinearDiffusion(I, 50, 0.05, 0.1)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].imshow(I, cmap='gray')
axes[0].set_title("Original")

axes[1].imshow(filtered_1, cmap='gray')
axes[1].set_title("Filtered 0.05dt 10itr")

axes[2].imshow(filtered_2, cmap='gray')
axes[2].set_title("Filtered 0.05dt 50itr")



plt.tight_layout()
plt.show()


