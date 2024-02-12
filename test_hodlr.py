import george
import numpy as np
import matplotlib.pyplot as pl


np.random.seed(1234)
x = np.sort(np.random.uniform(0, 10, 50000))
yerr = 0.1 * np.ones_like(x)
y = np.sin(x)

from george import kernels
kernel = np.var(y) * kernels.ExpSquaredKernel(1.0)

import time

t0 = time.time()
gp_basic = george.GP(kernel)
gp_basic.compute(x[:1000], yerr[:1000])
#print(gp_basic.log_likelihood(y[:1000]))
print(f"basic solver={time.time()-t0}")


t0 = time.time()
gp_hodlr = george.GP(kernel, solver=george.HODLRSolver, seed=42)
gp_hodlr.compute(x[:1000], yerr[:1000])
#print(gp_hodlr.log_likelihood(y[:1000]))
print(f"hodlr solver={time.time()-t0}")

import time

ns = np.array([50, 100, 200, 500, 1000, 5000, 10000, 50000], dtype=int)
t_basic = np.nan + np.zeros(len(ns))
t_hodlr = np.nan + np.zeros(len(ns))
t_gpy = np.nan + np.zeros(len(ns))
t_skl = np.nan + np.zeros(len(ns))
for i, n in enumerate(ns):
    # Time the HODLR solver.
    best = np.inf
    for _ in range(100000 // n):
        strt = time.time()
        gp_hodlr.compute(x[:n], yerr[:n])
        gp_hodlr.log_likelihood(y[:n])
        dt = time.time() - strt
        if dt < best:
            best = dt
    t_hodlr[i] = best

    # Time the basic solver.
    best = np.inf
    for _ in range(10000 // n):
        strt = time.time()
        gp_basic.compute(x[:n], yerr[:n])
        gp_basic.log_likelihood(y[:n])
        dt = time.time() - strt
        if dt < best:
            best = dt
    t_basic[i] = best

    # Compare to the proposed scikit-learn interface.
    #best = np.inf
    #if n <= 10000:
    #    gp_skl = GaussianProcessRegressor(kernel_skl,
    #                                      alpha=yerr[:n]**2,
    #                                      optimizer=None,
    #                                      copy_X_train=False)
    #    gp_skl.fit(x[:n, None], y[:n])
    #    for _ in range(10000 // n):
    #        strt = time.time()
    #        gp_skl.log_marginal_likelihood(kernel_skl.theta)
    #        dt = time.time() - strt
    #        if dt < best:
    #            best = dt
    #t_skl[i] = best

    # Compare to GPy.
    #best = np.inf
    #for _ in range(5000 // n):
    #    kernel_gpy = GPy.kern.RBF(input_dim=1, variance=np.var(y), lengthscale=1.#)
   #     strt = time.time()
   #     gp_gpy = GPy.models.GPRegression(x[:n, None], y[:n, None], kernel_gpy)
   #     gp_gpy['.*Gaussian_noise'] = yerr[0]**2
   #     gp_gpy.log_likelihood()
   #     dt = time.time() - strt
   #     if dt < best:
   #         best = dt
   # t_gpy[i] = best



#pl.loglog(ns, t_gpy, "-o", label="GPy")
#pl.loglog(ns, t_skl, "-o", label="sklearn")
pl.loglog(ns, t_basic, "-o", label="basic")
pl.loglog(ns, t_hodlr, "-o", label="HODLR")
pl.xlim(30, 80000)
pl.ylim(1.1e-4, 50.)
pl.xlabel("number of datapoints")
pl.ylabel("time [seconds]")
pl.legend(loc=2, fontsize=16);


#y = gp_hodlr.sample()
#print(y)

pl.show()
