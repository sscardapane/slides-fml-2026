import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import linear_model
colors = ['red', 'blue']

# General parameters: our dataset is generated from two
# overlapping Gaussians, one for each class.
mu_1 = 1.5
sigma_1 = 1.0
mu_2 = 6.0
sigma_2 = 1.5
N = 100

# Plot the two Gaussians
x_plot = np.linspace(-2, 10, 1000)
gaussian_1_plot = norm.pdf(x_plot, loc=mu_1, scale=sigma_1)
gaussian_2_plot = norm.pdf(x_plot, loc=mu_2, scale=sigma_2)

# Sample a few elements from the two classes
X = np.hstack((norm.rvs(size=(N,), loc=mu_1, scale=sigma_1),
             norm.rvs(size=(N,), loc=mu_2, scale=sigma_2))).reshape(2*N, 1)
y = np.hstack((np.zeros(N), np.ones(N)))

# Train a logistic regression model
logreg = linear_model.LogisticRegression().fit(X, y)
ypred = logreg.predict_proba(x_plot.reshape(-1, 1))

# Plot the dataset
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
plt.show()

# Plot the decision boundary of the model
plt.figure()
plt.plot(x_plot, ypred[:, 0], color=colors[0], label='$p(y_1 \mid x)$')
plt.plot(x_plot, ypred[:, 1], color=colors[1], label='$p(y_2 \mid x)$')
plt.xlabel('x')
plt.ylabel('$p(y \mid x)$')
plt.box(on=True)
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# Define an asymmetric cost matrix
C = np.asarray([[0, 1], [10, 0]])

cost_class_1 = ypred[:, 0]*C[0, 0] + ypred[:, 1]*C[0, 1]
cost_class_2 = ypred[:, 0]*C[1, 0] + ypred[:, 1]*C[1, 1]
intersection = np.argmin(np.square(cost_class_1 - cost_class_2))

# Plot the adjusted decision boundary
plt.figure()
plt.plot(x_plot, cost_class_1, color=colors[0], label='Cost (class 1)')
plt.plot(x_plot, cost_class_2, color=colors[1], label='Cost (class 2)')
plt.plot([x_plot[intersection], x_plot[intersection]], [0, 1], '--')
plt.plot('x')
plt.ylabel('Cost')
plt.box(on=True)
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()