import numpy as np
import scipy
import matplotlib.pyplot as plt
import skimage
import vapory

def generateDisks(nb_disks, S, Rmax):
    centers = np.random.randint(S, size=(nb_disks, 2))
    radii = Rmax * np.random.rand(nb_disks)

    N = 1000
    x = np.linspace(0, S, 1000)
    y = np.linspace(0, S, 1000)
    X, Y = np.meshgrid(x, y)
    I = np.zeros((N, N))
    for i in range(nb_disks):
        I2 = (X-centers[i, 0])**2 + (Y-centers[i, 1]) ** 2 <= radii[i]**2
        I = np.logical_or(I, I2)

    return I

def fractionalAreaDeterministic(Z):
    A = np.sum(Z > 0)
    return float(A)/(np.size(Z))

def fractionalAreaStochastic(nb_probes, Z):
    probes = np.random.randint(Z.shape[0], size=(nb_probes, 2))
    count = np.sum(Z[probes[:, 0], probes[:, 1]])
    return float(count)/nb_probes

def compareFractionalArea(Z):
    deterministic = fractionalAreaDeterministic(Z)
    stochastic = fractionalAreaStochastic(1500, Z)
    print(f"Aa deterministic: {deterministic}")
    print(f"Aa stochastic: {stochastic}")

def lengthPerAreaStochastic(Z):
    probes = np.zeros(Z.shape)
    probes[20:-20:10, 20:-20] = 1
    lines = Z.astype(int) * probes

    h = np.array([[1, -1, 0]])
    points = scipy.signal.convolve2d(lines, h, mode='same')
    nb_lines = np.sum(lines)
    nb_points = np.sum(np.abs(points))
    PL = float(nb_points)/nb_lines
    return points, np.pi/2*PL

def lengthPerAreaDeterministic(Z):
    perim = skimage.measure.perimeter(Z.astype(int), 8)
    LA = perim / np.sum(Z)
    return LA

def compareLengthPerArea(Z):
    _, PL = lengthPerAreaStochastic(Z)
    LA = lengthPerAreaDeterministic(Z)
    print(f"PL stochastic: {PL}")
    print(f"LA: {LA}")

def renderPopSpheres():
    nb_spheres = 50
    R = 5.
    centers = np.random.randint(100, size=(nb_spheres, 3))
    radius = np.random.randn(nb_spheres) * R + R
    couleurs = np.random.randint(255, size=(nb_spheres, 4))/255.

    camera = vapory.Camera('location', [150, 150, 150], 'look_at', [0, 0, 0])
    bg = vapory.Background('color', [1, 1, 1])
    light = vapory.LightSource([100, 100, 100], 'color', [1, 1, 1])
    light3 = vapory.LightSource([0, 0, 0], 'color', [1, 1, 1])
    light2 = vapory.LightSource([50, 50, 50], 'color', [1, 1, 1])

    obj = [light, light2, light3, bg]
    for i in range(nb_spheres):
        sphere = vapory.Sphere(centers[i, ], radius[i], vapory.Texture(vapory.Finish(
            'ambient', 0, 'reflection', 0, 'specular', 0, 'diffuse', 1), vapory.Pigment('color', couleurs[i, ])))
        obj.append(sphere)
    scene = vapory.Scene(camera, objects=obj)
    scene.render("./resources/spheres.png", width=3000, height=3000)

renderPopSpheres()
#
# Z = generateDisks(50, 1000, 50)
# compareFractionalArea(Z)
# compareLengthPerArea(Z)
# plt.imshow(Z)
# points, _ = lengthPerAreaStochastic(Z)
# P = np.where(np.abs(points)==1)
# plt.plot(P[1], P[0], '+')
# plt.show()

