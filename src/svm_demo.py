import numpy as np
import matplotlib.pyplot as plt

# Separating hyperplane parameters
m = 0.5
b = 0

# Generate data
x_pos = np.random.randn(10) - 2
x_neg = np.random.randn(10) + 2
y_pos = np.random.randn(10) + 2
y_neg = np.random.randn(10) - 2

# Generate separating hyperplane
x_hp = np.linspace(-5, 5, 10)
y_hp = m * x_hp + b

y_hp2 = (m + 0.3) * x_hp + (b - 0.3)
y_hp3 = (m - 0.3) * x_hp + (b + 0.3)

plt.title('Support Vector Machine Separating Hyperplane')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.scatter(x_pos, y_pos)
plt.scatter(x_neg, y_neg, color='red')
plt.plot(x_hp, y_hp)
plt.plot(x_hp, y_hp2)
plt.plot(x_hp, y_hp3)
plt.legend(['Optimal hyperplane', 'Alternative hyperplane 1', 'Alternative hyperplane 2'])

plt.show()
