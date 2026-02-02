
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from pylab import rcParams
import brewer2mpl
from matplotlib.colors import ListedColormap

font_size = 9

# Set parameters for plotting
params = {
   'axes.labelsize': font_size,
   'axes.linewidth': 1,
   'font.size': font_size,
   'legend.fontsize': font_size-2,
   'xtick.labelsize': font_size,
   'xtick.major.size': 2,
   'ytick.labelsize': font_size,
   'ytick.major.size': 2,
   'text.usetex': True,
   'figure.figsize': [4*0.9,3*0.9],
}
rcParams.update(params)

# Get a colors matrix
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors

np.random.seed(1)

N = 25

X1 = np.random.multivariate_normal(np.asarray([2, 2]), np.asarray([[1, 0.3], [0.3, 1]]), N)
X2 = np.random.multivariate_normal(np.asarray([0, 0]), np.asarray([[1, 1.5], [0, 1]]), N)

clf = neighbors.KNeighborsClassifier(30, metric='minkowski')
clf.fit(np.vstack((X1, X2)), np.hstack((np.zeros(N), np.ones(N))))

plt.figure()

x_min, x_max = X2[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X1[:,0], X1[:,1], marker='o', color=colors[0])
plt.scatter(X2[:,0], X2[:,1], marker='s', color=colors[1])

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])

plt.box(on=True)
plt.tight_layout()
plt.show()

plt.savefig('1nn-3.pdf', format='pdf',bbox_inches='tight', pad_inches=0, dpi=100)