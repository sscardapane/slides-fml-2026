
import numpy as np
import matplotlib.pyplot as plt
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
   'text.usetex': False,
   'figure.figsize': [4*0.9,3*0.9],
}
rcParams.update(params)

# Get a colors matrix
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors

np.random.seed(1)

N = 25
I = 3
C = 1

X1 = np.random.multivariate_normal(np.asarray([4, 4]), np.asarray([[1, 0], [0, 1]]), N)
X2 = np.random.multivariate_normal(np.asarray([0, 0]), np.asarray([[1, 0], [0, 1]]), N)
X = np.hstack((np.vstack((X1, X2)), np.ones((N*2, 1))))

y = np.vstack((np.zeros(((N, 1))), np.ones((N, 1))))

w = np.random.randn(3)

sigmoid = lambda s: 1.0/(1.0+np.exp(-s))

for k in np.arange(0, I):
    
    p = sigmoid(X.dot(w))
    W = np.diag(p*(1.0-p))
    w = w + np.linalg.solve(X.T.dot(W).dot(X) + C*np.eye(3), X.T.dot(y-p.reshape(-1, +1)).reshape(-1)).reshape(-1)

x_min, x_max = X2[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

xy_plot = np.c_[xx.ravel(), yy.ravel()]
xy_plot = np.hstack((xy_plot, np.ones((xy_plot.shape[0], 1))))
Z = np.round(sigmoid(xy_plot.dot(w)))

plt.figure()

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X1[:,0], X1[:,1], marker='o', color=colors[0])
plt.scatter(X2[:,0], X2[:,1], marker='s', color=colors[1])

plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))

plt.box(on=True)
plt.grid()
plt.tight_layout()
plt.show()

plt.savefig('logistic_regression_2.pdf', format='pdf',bbox_inches='tight', pad_inches=0)