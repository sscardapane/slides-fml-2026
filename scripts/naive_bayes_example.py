import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(1)

def gaussian(x, mu, sigma):
    # A Gaussian centered in mu and with variance sigma
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.0) / 2)
    )

# Generate a dataset of N points composed of a mixture of two Gaussians
N = 25
X1 = np.random.multivariate_normal(np.asarray([4, 4]), np.asarray([[1, 0], [0, 1]]), N)
X2 = np.random.multivariate_normal(np.asarray([0, 0]), np.asarray([[1.3, 0.1], [1.2, 3.1]]), N)

# The first N points (first Gaussian) have class 0, the remaining N points
# (second Gaussian) have class 1
y = np.vstack((np.zeros(((N, 1))), np.ones((N, 1))))

# Compute empirical means
mu1 = np.mean(X1, axis=0)
mu2 = np.mean(X2, axis=0)

# Compute empirical variances
sigma1 = np.var(X1, axis=0)
sigma2 = np.var(X2, axis=0)

plt.figure()

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# Plot observations
plt.scatter(X1[:,0], X1[:,1], marker='o')
plt.scatter(X2[:,0], X2[:,1], marker='s')

# We compute the predicted values of the two Gaussians according to Naive Bayes
x_plot = np.linspace(-2, 6, 100)
x, y = np.meshgrid(x_plot, x_plot)
x, y = x.ravel(), y.ravel()
z1 = gaussian(x, mu1[0], sigma1[0])*gaussian(y, mu1[1], sigma1[1])
z2 = gaussian(x, mu2[0], sigma2[0])*gaussian(y, mu2[1], sigma2[1])
plt.contourf(x_plot, x_plot, z1.reshape(100, 100), cmap='Blues', alpha=0.2)
plt.contourf(x_plot, x_plot, z2.reshape(100, 100), cmap='Reds', alpha=0.2)

plt.show()