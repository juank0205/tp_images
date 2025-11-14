import skimage.io
import numpy as np
import matplotlib.pyplot as plt

def linearDiffusion(I, nb_iterations: int, dt: float):
    Z = I.copy()
    for _ in range(nb_iterations):
        gN = np.roll(Z, 1, axis=0) - Z
        gE = np.roll(Z, -1, axis=1) - Z
        gS = np.roll(Z, -1, axis=0) - Z
        gW = np.roll(Z, 1, axis=1) - Z
        Z = Z + dt * (gN + gE + gS + gW)
    return Z


I = skimage.io.imread("./images/cerveau.png")/255
filtered_1 = linearDiffusion(I, 50, 0.05)
filtered_2 = linearDiffusion(I, 200, 0.05)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].imshow(I, cmap='gray')
axes[0].set_title("Original")

axes[1].imshow(filtered_1, cmap='gray')
axes[1].set_title("Filtered 0.05dt 50itr")

axes[2].imshow(filtered_2, cmap='gray')
axes[2].set_title("Filtered 0.05dt 200itr")



plt.tight_layout()
plt.show()


