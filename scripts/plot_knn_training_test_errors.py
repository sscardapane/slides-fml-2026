
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
N_test = 20

X1 = np.random.multivariate_normal(np.asarray([2, 2]), np.asarray([[1, 0.3], [0.3, 1]]), N)
X2 = np.random.multivariate_normal(np.asarray([0, 0]), np.asarray([[1, 1.5], [0, 1]]), N)

X1_test = np.random.multivariate_normal(np.asarray([2, 2]), np.asarray([[1, 0.3], [0.3, 1]]), N_test)
X2_test = np.random.multivariate_normal(np.asarray([0, 0]), np.asarray([[1, 1.5], [0, 1]]), N_test)

trn_error = np.zeros(11)
tst_error = np.zeros(11)

for (i,k) in zip(np.arange(0,11), np.arange(1, 51, 5)):

    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(np.vstack((X1, X2)), np.hstack((np.zeros(N), np.ones(N))))

    trn_error[i] = clf.score(np.vstack((X1, X2)), np.hstack((np.zeros(N), np.ones(N))))
    tst_error[i] = clf.score(np.vstack((X1_test, X2_test)), np.hstack((np.zeros(N_test), np.ones(N_test))))


plt.figure()

plt.plot(trn_error, 'r')
plt.plot(tst_error, 'b')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.box(on=True)
plt.tight_layout()
plt.show()

#plt.savefig('1nn-3.pdf', format='pdf',bbox_inches='tight', pad_inches=0)