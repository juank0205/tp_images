import numpy as np
import matplotlib.pyplot as plt

def randomRadius(R, n):
    x = R * np.random.rand(n)
    r = np.sqrt(R**2 - x**2)

    probability = np.histogram(r, bins=1000)
    plt.plot(probability[1][:-1], probability[0]/n, linewidth=2)

def randomEndpoints(R, n):
    # t1 = 2*np.pi * np.random.rand(n)
    # t2 = 2*np.pi * np.random.rand(n)
    #
    # p1x = R * np.cos(t1)
    # p1y = R * np.sin(t1)
    # p2x = R * np.cos(t2)
    # p2y = R * np.sin(t2)
    #
    # r = []
    # for p1xi, p1yi, p2xi, p2yi in zip(p1x, p1y, p2x, p2y):
    #     d = np.sqrt((p1xi - p2xi)**2 + (p1yi - p2yi)**2)
    #     r.append(d)

    t = 2*np.pi * np.random.rand(n)
    dX = np.diff(R * np.cos(t))
    dY = np.diff(R * np.sin(t))
    r = 1./2 * np.sqrt(dX**2 + dY**2)

    probability = np.histogram(r, bins=1000)
    plt.plot(probability[1][:-1], probability[0]/n, linewidth=2)

def analyticValues(R, step):
    r2 = np.arange(0, R, step)
    probability = 1./R * r2 / np.sqrt(R**2-r2**2)
    probability = probability * R / 1000
    plt.scatter(r2, probability, 50)

def generatePointsOnSphere(R, n):
    points = np.random.rand(n, 3);
    norm = np.linalg.norm(points, axis=1)
    points = R * n / norm[:, None]
    return points


def endpointsSphere3(R, n):
    x = generatePointsOnSphere(R, n)
    y = generatePointsOnSphere(R, n)
    z = generatePointsOnSphere(R, n)

    u = y-x
    v = z-x

    normal = np.cross(u, v)


randomRadius(1, 10000000)
randomEndpoints(1, 10000000)
analyticValues(1, 0.05)
plt.show()
