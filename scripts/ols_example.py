import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Set the random seed
np.random.seed(1)

LINEAR = True # Set to False for the non-linear variant

# Number of points in the dataset
N = 25

def f_linear(x):
   # Linear function
   return 0.25 * x

def f_nonlinear(x):
   # Nonlinear function
   return 0.6*np.sin(np.sqrt(x)*3)**3 - 0.1*x + 0.3

f = f_linear if LINEAR else f_nonlinear

# Generate the dataset
X = np.random.rand(N, 1)*10
y = f(X) + np.random.randn(N, 1)*0.5  # Dataset with noise

# Add a rows of ones
X_ones = np.hstack((X, np.ones((N,1))))

# Closed-form solution
w = np.linalg.solve(X_ones.T @ X_ones, X_ones.T @  y)

# Closed form solution for sigma
sigma = ((y - X_ones @ w)**2).mean()

# Grid for plotting the true function
x_plot = np.arange(0, 10, 0.01)
y_plot = f(x_plot)

# We only need two points to plot OLS
x_ext = [0, 10]
y_ext = np.asarray([w[1,0], w[0,0]*x_ext[1] + w[1,0]])

plt.figure()
plt.grid(color="0.1")
plt.scatter(X, y, label='Observations')
plt.plot(x_plot, y_plot, label='True')
plt.plot(x_ext, y_ext, '-', label='OLS')
plt.fill_between(x_ext, y_ext - sigma,  y_ext+sigma, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(x_ext)
leg = plt.legend(loc='upper right')
plt.show()