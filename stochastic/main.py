from math import floor, sqrt
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def cond_poisson(density, xmin, xmax, ymin, ymax):
    x = xmin+(xmax-xmin)*np.random.rand(density)
    y = ymin+(ymax-ymin)*np.random.rand(density)
    return x, y

def neyman_scott(hchild, npar, xmin, xmax, ymin, ymax, r):
    nSons = scipy.stats.poisson.rvs(hchild, size=npar)
    aggx, aggy = cond_poisson(npar, xmin, xmax, ymin, ymax)
    childrenx, childreny = [], []
    for aggxi, aggyi, nSonsi in zip(aggx, aggy, nSons):
        aggx_children, aggy_children =cond_poisson(nSonsi, aggxi-r/2, aggxi+r/2, aggyi-r/2, aggyi+r/2)
        childrenx.append(aggx_children)
        childreny.append(aggy_children)
    return aggx, aggy, childrenx, childreny

def display_cond(x, y, title, ncols, nrows, index):
    plt.subplot(ncols, nrows, index)
    plt.plot(x, y, "+")
    plt.title(title)

def display_neyron(aggx, aggy, childx, childy):
    childrenx = np.concatenate(childx).tolist()
    childreny = np.concatenate(childy).tolist()
    plt.plot(aggx, aggy, "+")
    plt.plot(childrenx, childreny, "+")

def display_gibbs(x, y):
    plt.plot(x, y, "+")

def energyFunction(d):
    if d <= 15 and d > 0:
        return 50
    elif d <=30:
        return -10
    else:
        return 0

def euclideanDistance(x1, y1, x2, y2):
    # print(x1, x2, y1, y2)
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def energy(x, y, xk, yk, energyFunctionCallback):
    total_energy = 0
    for xi, yi in zip(x, y):
        total_energy += energyFunctionCallback(euclideanDistance(xk, xi, yk, yi))
    return total_energy

def gibbs(iPoints, iterations, candidates):
    x, y = cond_poisson(iPoints, -1000, 1000, -1000, 1000)
    display_cond(x, y, "Original", 1, 2, 1)
    print(len(x))
    for _ in range(iterations):
        chosen_one = floor(np.random.rand() * iPoints)
        chosen_x, chosen_y = x[chosen_one], y[chosen_one]
        min_energy = energy(x, y, chosen_x, chosen_y, energyFunction)
        x = np.delete(x, chosen_one)
        y = np.delete(y, chosen_one)
        for _ in range(candidates):
            candidate_x, candidate_y = cond_poisson(1, -1000, 1000, -1000, 1000) 
            candidate_x = candidate_x[0]
            candidate_y = candidate_y[0]
            candidate_energy = energy(x, y, candidate_x, candidate_y, energyFunction)
            if candidate_energy <= min_energy:
                min_energy = candidate_energy
                chosen_x = candidate_x
                chosen_y = candidate_y
        x = np.append(x, chosen_x)
        y = np.append(y, chosen_y)

    return x, y

x, y = gibbs(1000, 1, 4)
display_cond(x, y, "Gibbs", 1, 2, 2)
plt.show()

# aggx, aggy, childrenx, childreny = neyman_scott(5, 10, -10, 10, -10, 10, 2)
# display_neyron(aggx, aggy, childrenx, childreny)
# plt.show()

# x, y = cond_poisson(200, -10, 10, -10, 10)
# display_figures(x, y, "Distribution 1", 2, 2, 1)
# x, y = cond_poisson(200, -10, 10, -10, 10)
# display_figures(x, y, "Distribution 2", 2, 2, 2)
# x, y = cond_poisson(200, -10, 10, -10, 10)
# display_figures(x, y, "Distribution 3", 2, 2, 3)
# x, y = cond_poisson(200, -10, 10, -10, 10)
# display_figures(x, y, "Distribution 4", 2, 2, 4)
# plt.show()

