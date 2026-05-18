# -*- coding: utf-8 -*-

from sklearn import datasets, preprocessing
import matplotlib.pyplot as plt
from pylab import rcParams
import brewer2mpl

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
bmap = brewer2mpl.get_map('Set1', 'qualitative', 4)
colors = bmap.mpl_colors

# Load dataset
X = datasets.load_boston()['data']
X_minmax = preprocessing.MinMaxScaler().fit_transform(X)
X_normal = preprocessing.StandardScaler().fit_transform(X)
X_robust = preprocessing.RobustScaler().fit_transform(X)

plt.figure()
plt.grid()
plt.hist(X[:,2], color=colors[3])
plt.xlabel('Value')
plt.ylabel('# values')
plt.box(on=True)
plt.tight_layout()
plt.show()
plt.savefig('feature_normalization_original.pdf', format='pdf',bbox_inches='tight', pad_inches=0)

plt.figure()
plt.grid()
plt.hist(X_minmax[:,2], color=colors[0])
plt.xlabel('Value')
plt.ylabel('# values')
plt.box(on=True)
plt.tight_layout()
plt.show()
plt.savefig('feature_normalization_minmax.pdf', format='pdf',bbox_inches='tight', pad_inches=0)

plt.figure()
plt.grid()
plt.hist(X_normal[:,2], color=colors[1])
plt.xlabel('Value')
plt.ylabel('# values')
plt.box(on=True)
plt.tight_layout()
plt.show()
plt.savefig('feature_normalization_normal.pdf', format='pdf',bbox_inches='tight', pad_inches=0)

plt.figure()
plt.grid()
plt.hist(X_robust[:,2], color=colors[2])
plt.xlabel('Value')
plt.ylabel('# values')
plt.box(on=True)
plt.tight_layout()
plt.show()
plt.savefig('feature_normalization_robust.pdf', format='pdf',bbox_inches='tight', pad_inches=0)