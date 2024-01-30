import numpy as np
from matplotlib import pylab as plt

l = 0.01
sigma = 1.0
N = 1000
X = np.linspace(-1, 1, N)
K = sigma**2 * np.exp(- np.abs(np.subtract.outer(X,X))**2 / (2 * l**2))
eps = 1e-4 * np.linalg.norm(K, 2)
K = K + eps * np.eye(N)

L = np.linalg.cholesky(K)
z = np.random.randn(N, 1)

y = np.dot(L,z)

fig, ax = plt.subplots()
ax.plot(X,y,'-b')
ax.grid()
plt.show()
