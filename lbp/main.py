import glob
import pandas as pd
import seaborn as sn
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.cluster import KMeans

def LBP(I):
    result = np.zeros(I.shape, dtype=int)
    for i in range(1, I.shape[0] - 1):      # rows (y)
        for j in range(1, I.shape[1] - 1):  # columns (x)
            W = I[i-1:i+2, j-1:j+2]
            W = W >= I[i, j]
            bits = [
                W[0,0],  # top-left
                W[0,1],  # top-middle
                W[0,2],  # top-right
                W[1,0],  # middle-left
                W[1,2],  # midle-right
                W[2,0],  # bottom-left
                W[2,1],  # bottom-middle
                W[2,2],  # bottom-right
            ]

            result[i][j] = sum(b * (2**i) for i, b in enumerate(bits))

    hist, bins = np.histogram(result[1:-1, 1:-1], bins=256, density=True)
    return hist
            
# I = skimage.io.imread("images/Metal.1.bmp")
# lbp = LBP(I)
# plt.plot(lbp)
# plt.ylim(0, 0.08)  # set max to 0.1
# plt.xlabel('Pattern value')
# plt.ylabel('Frequency')
# plt.show()

def plot_dists(dists, classes, cmap=plt.cm.Blues):
    """
    Plot matrix of distances
    dists: all computed distances
    classes: labels to be used
    cmap: colormap

    returns: figure that can be used for pdf export
    """
    df_cm = pd.DataFrame(dists, index=classes, columns=classes)

    fig = plt.figure()
    sn.set(font_scale=.8)
    sn.heatmap(df_cm, annot=True, cmap=cmap, fmt='.2f')

    return fig

def process_image(file):
    """Read an image, compute its LBP, return both for plotting"""
    I = skimage.io.imread(file)
    lbp = LBP(I)
    return I, lbp, file

classes = ["Metal", "Sand", "Terrain"]
hh = []
names = []
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for idx, c in enumerate(classes):
    files = sorted(glob.glob(f'./images/{c}*.bmp'))

    # Use ThreadPoolExecutor to process all images in this class in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, f) for f in files]

        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            I, lbp, file = future.result()
            results.append((I, lbp))
            names.append(os.path.basename(file))
            hh.append(lbp)

    # --- Top row: show a sample image (first one) ---
    axes[0, idx].imshow(results[0][0], cmap='gray')
    axes[0, idx].set_title(c)
    axes[0, idx].axis('off')

    # --- Bottom row: plot histograms for all images ---
    for _, lbp in results:
        axes[1, idx].plot(lbp, alpha=0.5)
    axes[1, idx].set_ylim(0, 0.08)
    axes[1, idx].set_xlabel('Pattern value')
    axes[1, idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# kmeans clustering
n = 3
k_means = KMeans(init='k-means++', n_clusters=n, n_init=10)
k_means.fit(hh)
print(k_means.labels_)

n = len(hh)
dists = np.zeros((n, n))
for i in np.arange(n):
    for j in np.arange(n):
        dists[i, j] = np.sum(np.abs(hh[i]-hh[j]))

fig = plot_dists(dists, names)
plt.show()
