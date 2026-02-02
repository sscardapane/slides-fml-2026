
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

N = 50
X = np.random.multivariate_normal(np.asarray([1, 1]), np.asarray([[25.0, 0], [0, 0.1]]), N)


plt.figure()

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

plt.scatter(X[:,0], X[:,1], marker='o', color=colors[1])

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])

plt.grid()
plt.box(on=True)
plt.tight_layout()
plt.show()

plt.savefig('2d_points.pdf', format='pdf',bbox_inches='tight', pad_inches=0)