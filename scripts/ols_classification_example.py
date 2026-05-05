import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(1)

# Define the dataset
N = 25
X1 = np.random.multivariate_normal(np.asarray([4, 4]), np.asarray([[1, 0], [0, 1]]), N)
X2 = np.random.multivariate_normal(np.asarray([0, 0]), np.asarray([[1, 0], [0, 1]]), N)
X3 = np.random.multivariate_normal(np.asarray([8, 8]), np.asarray([[1, 0], [0, 1]]), N)
X = np.hstack((np.vstack((X1, X2, X3)), np.ones((N*3, 1))))

# Generate outputs in a one-hot encoded form
y = np.hstack((np.zeros((N,)), 2.0*np.ones((N,)), np.ones((N,)))).astype(np.int32)
Y = np.eye(3)[y]

# Compute least-squares solution
W = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

plt.figure()

# Grid for plotting
x_min, x_max = X2[:, 0].min() - 1, X3[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X3[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Compute model's predictions
xy_plot = np.c_[xx.ravel(), yy.ravel()]
xy_plot = np.hstack((xy_plot, np.ones((xy_plot.shape[0], 1))))
Z = np.argmax(xy_plot.dot(W), axis=1)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X1[:,0], X1[:,1], marker='o')
plt.scatter(X2[:,0], X2[:,1], marker='s')
plt.scatter(X3[:,0], X3[:,1], marker='x')

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])

plt.box(on=True)
plt.grid()
plt.tight_layout()
plt.show()