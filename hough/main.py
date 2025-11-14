import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import peak_local_max

def hough(I):
    edges = cv2.Canny(I, 100, 200)  
    theta = np.arange(0, np.pi, 0.01)
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    rho_max = np.hypot(I.shape[0], I.shape[1])
    rho = np.arange(-rho_max, rho_max, 1)
    H = np.zeros([rho.size, theta.size])
    for x in range(I.shape[0]):
        for y in range(I.shape[1]):
            if (edges[x, y] != 0):
                R = x*cosTheta + y*sinTheta
                R = np.round(R + rho.size/2).astype(int)
                H[R, range(theta.size)] += 1

    # plt.imshow(H, extent=[0, np.pi, -rho_max, rho_max], aspect='auto')
            
    G = cv2.GaussianBlur(H, (5, 5), 5)
    maxima = peak_local_max(H, 5, threshold_abs=150, num_peaks=5)
    # plt.figure()
    # plt.imshow(G, aspect='auto')
    # plt.scatter(maxima[:, 1], maxima[:, 0], c='r')
    # plt.show()

    for i_rho, i_theta in maxima:
        a = np.cos(theta[i_theta])
        b = np.sin(theta[i_theta])
        y0 = a * rho[i_rho]
        x0 = b * rho[i_rho]
        y1 = int(y0 + 1000*(-b))
        x1 = int(x0 + 1000*(a))
        y2 = int(y0 - 1000*(-b))
        x2 = int(x0 - 1000*(a))

        cv2.line(I, (x1, y1), (x2, y2), (0, 0, 255), 2)
    plt.imshow(I)
    plt.show()
            


I = cv2.imread("TestPR46.png")
H = hough(I)

