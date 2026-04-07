import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap

np.random.seed(1)

# Number of points
N = 25

# Dataset
X1 = np.random.multivariate_normal(np.asarray([2, 2]), np.asarray([[1, 0.3], [0.3, 1]]), N)
X2 = np.random.multivariate_normal(np.asarray([0, 0]), np.asarray([[1, 1.5], [0, 1]]), N)

# Note: sklearn is introduced later in the course
clf = neighbors.KNeighborsClassifier(1, metric='euclidean')
clf.fit(np.vstack((X1, X2)), np.hstack((np.zeros(N), np.ones(N))))

plt.figure()

# Get ranges for plotting
x_min, x_max = X2[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X1[:, 1].max() + 1

# Grid for plotting the decision boundaries
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predictions of the k-NN
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot the decision boundaries
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the points
plt.scatter(X1[:,0], X1[:,1], marker='o')
plt.scatter(X2[:,0], X2[:,1], marker='s')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])

plt.box(on=True)
plt.tight_layout()
plt.show()