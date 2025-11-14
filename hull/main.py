from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

def generatePoints(n: int, windowSize: int) -> Tuple[np.ndarray, np.ndarray]:
    Ix = np.random.rand(n) * windowSize
    Iy = np.random.rand(n) * windowSize
    return  Ix, Iy

def getLowerY(Iy) -> np.int64:
    return np.argmin(Iy)

def sortByAngle(S: Tuple[np.ndarray, np.ndarray], P: Tuple[float, float]) -> np.ndarray:
    x, y = S
    hypothenuses = np.sqrt((x-P[0])**2 + (y-P[1])**2)
    adjacents = x - P[0]
    cosinuses = -adjacents / hypothenuses
    indices = np.argsort(cosinuses)

    return indices

def constructL(P: Tuple[float, float], indices: np.ndarray, S: Tuple[np.ndarray, np.ndarray]) -> list:
    S_sorted = (S[0][indices], S[1][indices])
    L = (np.concatenate([[P[0]], S_sorted[0], [P[0]]]).tolist(), np.concatenate([[P[1]], S_sorted[1], [P[1]]]).tolist())
    L = [[xi, yi] for xi, yi in zip(*L)]
    return L

def ccw(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    return (p2[0]-p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def convexHull(Ix: np.ndarray, Iy: np.ndarray):
    Pindex = getLowerY(Iy)
    P = (Ix[Pindex], Iy[Pindex])
    S = (np.delete(Ix, Pindex), np.delete(Iy, Pindex))
    angleIndices = sortByAngle(S, P)
    L = constructL(P, angleIndices, S)
    
    hull = [] # stackini
    hull.append(L.pop(0))
    hull.append(L.pop(0))

    for p in L:
        while len(hull)>=2 and ccw(hull[-1], hull[-2], p)>0:
            hull.pop()
        hull.append(p)

    return hull

def displayPointsAndHull(pointsX, pointsY, hull=None):
    """
    Display all points and optionally the convex hull polygon.
    """
    plt.figure()

    # scatter all points
    plt.scatter(pointsX, pointsY, color='C0', label='Points')

    # draw hull polygon if provided
    if hull is not None:
        hull = np.array(hull)
        # close the polygon by repeating the first point at the end
        hull_closed = np.vstack([hull, hull[0]])
        plt.plot(hull_closed[:, 0], hull_closed[:, 1], 'C2-', label='Hull')
    plt.show()


Ix, Iy = generatePoints(20, 500)
hull  =convexHull(Ix, Iy)
displayPointsAndHull(Ix, Iy, hull)

