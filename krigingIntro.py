import numpy as np
from matplotlib import pylab as plt
import sklearn


X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
xt = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]).reshape(-1, 1)
xt = np.array([0.2, 1.1, 2.7, 4.5, 5.7]).reshape(-1, 1)
yt = np.sin(xt).reshape(-1, 1)
print(np.mean(yt))
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    ExpSineSquared,
    ConstantKernel,
    RBF,
)

kernel = ConstantKernel() * RBF()
#kernel = RBF()

gpr = GaussianProcessRegressor(kernel, normalize_y=True,n_restarts_optimizer=5,).fit(xt, yt)
#gpr = GaussianProcessRegressor(normalize_y=True).fit(xt, yt)
y_mean, y_std = gpr.predict(X, return_std=True)
ys = gpr.sample_y(X, 5)


# graphic
fig, ax = plt.subplots()
ax.plot(X[:, 0], y_mean.ravel(), "-b", label="sklearn predictor")
ax.fill_between(
    X[:, 0],
    y_mean.ravel() - 1.96 * y_std.ravel(),
    y_mean.ravel() + 1.96 * y_std.ravel(),
    color="b",
    alpha=0.25,
    label=r"$[-3\sigma, +3\sigma]$ confidence interval",
)
ax.plot(X[:, 0], np.sin(X[:, 0]), "--r", label="exact solution")
ax.plot(xt, yt, "or", label="data")
ax.plot(X[:, 0], ys[:, 0, 0], "--k", label="one conditioned realization")
ax.grid()


#kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
#    1.0, length_scale_bounds="fixed"
#)

print(gpr.get_params())
print(gpr.kernel_)
print(gpr.kernel_.theta)

l = 1.0
sigma = 1.0
Ktt = np.exp(-np.abs(np.subtract.outer(xt.ravel(), xt.ravel())) ** 2 / (2.0 * l ** 2))
Ltt = np.linalg.cholesky(Ktt)

z = np.linalg.solve(Ktt, (yt - np.mean(yt)))

Kn = np.exp(-np.abs(np.subtract.outer(xt.ravel(), X.ravel())) ** 2 / (2.0 * l ** 2))
esp = np.dot(Kn.T, z)
ax.plot(X[:, 0], esp + np.mean(yt), "--", color="orange", label=r"$\mathbb{E}[Y]$")
ax.legend()
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$Z_t$")
ax.set_title(r"Simple Gaussian Process Regression exemple: $x \mapsto \sin(x)$")




def build_covariance_matrix(xt, params):
    sigma, ll = params
    return sigma**2 *  np.exp(-np.abs(np.subtract.outer(xt.ravel(), xt.ravel())) ** 2 / (2.0 * ll ** 2))


def log_likelihood(y, K):
    """log likelihood function"""
    n = 5
    ll = -0.5 * np.dot(y.T, np.linalg.solve(K, y)) - 0.5 * np.log(np.abs(np.linalg.det(K))) - n / 2 * np.log(2*np.pi)
    return ll


ll = log_likelihood(yt, Ktt)
print(ll)
print(np.info(Ktt))


def foo(params):
    Ktt = build_covariance_matrix(xt, params)
    return -log_likelihood(yt, Ktt)
    
from scipy.optimize import minimize

bnds = ((0, None), (0, None))
x0 = np.array([1.0, 1.0])
print(foo(x0))
res = minimize(foo, x0, bounds=bnds,method='L-BFGS-B')
print(res)

Ktt = build_covariance_matrix(xt, res.x)
z = np.linalg.solve(Ktt, (yt - np.mean(yt)))
Knn = res.x[0]**2 * np.exp(-np.abs(np.subtract.outer(xt.ravel(), X.ravel())) ** 2 / (2.0 * (res.x[1]) ** 2))
esp2 = np.dot(Knn.T, z)
ax.plot(X[:, 0], esp2 + np.mean(yt), "--", color="saddlebrown", label=r"$\mathbb{E}[Y]$")



plt.show()
