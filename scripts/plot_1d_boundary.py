import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import linear_model
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

# General parameters
mu_1 = 1.5
sigma_1 = 1.0
mu_2 = 6.0
sigma_2 = 1.5
N = 100

# Plot the two Gaussians
x_plot = np.linspace(-2, 10, 1000)
gaussian_1_plot = norm.pdf(x_plot, loc=mu_1, scale=sigma_1)
gaussian_2_plot = norm.pdf(x_plot, loc=mu_2, scale=sigma_2)

# Sample a few elements
X = np.hstack((norm.rvs(size=(N,), loc=mu_1, scale=sigma_1),
             norm.rvs(size=(N,), loc=mu_2, scale=sigma_2))).reshape(2*N, 1)
y = np.hstack((np.zeros(N), np.ones(N)))

# Train a logistic regression model
logreg = linear_model.LogisticRegression().fit(X, y)
ypred = logreg.predict_proba(x_plot.reshape(-1, 1))

plt.figure()
plt.plot(x_plot, gaussian_1_plot, color=colors[0], label='$p(x \mid y_1)$')
plt.scatter(X[:N, 0], np.zeros(N,), color=colors[0], label='Observations (class 1)')
plt.scatter(X[N:, 0], np.zeros(N,), color=colors[1], label='Observations (class 2)')
plt.plot(x_plot, gaussian_2_plot, color=colors[1], label='$p(x \mid y_2)$')
plt.xlabel('x')
plt.ylabel('$p(x)$')
plt.box(on=True)
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('decision_boundaries_1.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()

plt.figure()
plt.plot(x_plot, ypred[:, 0], color=colors[0], label='$p(y_1 \mid x)$')
plt.plot(x_plot, ypred[:, 1], color=colors[1], label='$p(y_2 \mid x)$')
plt.xlabel('x')
plt.ylabel('$p(y \mid x)$')
plt.box(on=True)
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('decision_boundaries_2.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()

C = np.asarray([[0, 1], [1, 0]])

cost_class_1 = ypred[:, 0]*C[0, 0] + ypred[:, 1]*C[0, 1]
cost_class_2 = ypred[:, 0]*C[1, 0] + ypred[:, 1]*C[1, 1]
intersection = np.argmin(np.square(cost_class_1 - cost_class_2))

plt.figure()
plt.plot(x_plot, cost_class_1, label='Cost (class 1)')
plt.plot(x_plot, cost_class_2, label='Cost (class 2)')
plt.plot([x_plot[intersection], x_plot[intersection]], [0, 1], '--')
plt.plot('x')
plt.ylabel('Cost')
plt.box(on=True)
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('decision_boundaries_3.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()

C = np.asarray([[0, 10], [1, 0]])

cost_class_1 = ypred[:, 0]*C[0, 0] + ypred[:, 1]*C[0, 1]
cost_class_2 = ypred[:, 0]*C[1, 0] + ypred[:, 1]*C[1, 1]
intersection = np.argmin(np.square(cost_class_1 - cost_class_2))

plt.figure()
plt.plot(x_plot, cost_class_1, label='Cost (class 1)')
plt.plot(x_plot, cost_class_2, label='Cost (class 2)')
plt.plot([x_plot[intersection], x_plot[intersection]], [0, 10], '--')
plt.plot('x')
plt.ylabel('Cost')
plt.box(on=True)
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('decision_boundaries_4.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()