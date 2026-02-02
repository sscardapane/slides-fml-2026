
import numpy as np
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
   'text.usetex': True,
   'figure.figsize': [4*0.9,3*0.9],
}
rcParams.update(params)

# Get a colors matrix
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors

np.random.seed(1)

x = np.arange(-5, 5, 0.01)

plt.figure()

plt.plot(x, np.square(x), color=colors[0])

plt.xlabel('Error')
plt.ylabel('Squared loss')

plt.grid()
plt.box(on=True)
plt.tight_layout()
plt.show()

plt.savefig('squared_loss.pdf', format='pdf',bbox_inches='tight', pad_inches=0)