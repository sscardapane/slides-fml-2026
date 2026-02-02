from numpy import linspace
import matplotlib.pyplot as plt

# Original function
f = lambda x: x**2 - 1.5*x

# Derivative (manual)
df = lambda x: 2*x - 1.5

# Linearization in 0.5
x = 0.5
f_linearized = lambda h: f(x) + df(x)*(h - x)

# Comparison
print(f(x + 0.01))
print(f_linearized(x + 0.01))

# For plotting (a grid of finely-spaced points over the x-axis)
xrange = linspace(-1, 1, 1000)

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(xrange, f(xrange), label='f(x)')
plt.plot(xrange, f_linearized(xrange), label='Linearized at 0.5')

ax.fill_between(xrange, f(xrange), f_linearized(xrange), facecolor='gray', alpha=0.1, interpolate=True)

plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')

plt.legend()
plt.grid(color="0.9", linestyle='--', linewidth=1)
plt.box(on=True)
plt.tight_layout()
plt.show()