
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
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
   'text.usetex': True,
   'figure.figsize': [4*0.9,3*0.9],
}
rcParams.update(params)

# Get a colors matrix
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors

np.random.seed(1)

N = 25

x = np.random.rand(N, 1)*10

y = 0.25*x + np.random.randn(N, 1)*0.5  #linear
#y = 0.6*np.sin(np.sqrt(x)*3)**3 - 0.1*x + 0.3

x_ones = np.hstack((x, np.ones((N,1))))
w = linalg.inv(x_ones.T.dot(x_ones)).dot(x_ones.T.dot(y))
sigma = ((y - x_ones @ w)**2).mean()

x_plot = np.arange(0, 10, 0.01)
#y_plot = 0.6*np.sin(np.sqrt(x_plot)*3)**3 - 0.1*x_plot + 0.3

x_ext = [0, 10]
y_ext = [w[1,0], w[1,0]+10*w[0,0]]

plt.figure()
plt.grid(color="0.1")
plt.scatter(x, y, label='Observations', color=colors[1])
plt.plot(x, w[1] + w[0]*x, '-', label='OLS', color=colors[0])
plt.fill_between(x_ext, y_ext - sigma,  y_ext+sigma, color=colors[0], alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, np.max(x)])
leg = plt.legend(loc='upper right')
fr = leg.get_frame()
fr.set_lw(0.5)
plt.box(on=True)
plt.tight_layout()
plt.savefig('./OLS_linear_data_sigma.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()