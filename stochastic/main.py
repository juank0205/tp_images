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

aggx, aggy, childrenx, childreny = neyman_scott(5, 10, -10, 10, -10, 10, 2)
display_neyron(aggx, aggy, childrenx, childreny)
plt.show()

# x, y = cond_poisson(200, -10, 10, -10, 10)
# display_figures(x, y, "Distribution 1", 2, 2, 1)
# x, y = cond_poisson(200, -10, 10, -10, 10)
# display_figures(x, y, "Distribution 2", 2, 2, 2)
# x, y = cond_poisson(200, -10, 10, -10, 10)
# display_figures(x, y, "Distribution 3", 2, 2, 3)
# x, y = cond_poisson(200, -10, 10, -10, 10)
# display_figures(x, y, "Distribution 4", 2, 2, 4)
# plt.show()

