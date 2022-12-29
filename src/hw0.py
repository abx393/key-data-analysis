import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0, 2, 4], [2, 4, 2], [3, 3, 1]])
A_inv = np.linalg.inv(A)
#print(A_inv)

b = np.array([[-2], [-2], [-4]])
c = np.array([[1], [1], [1]])

#print('A^-1 b: \n', np.matmul(A_inv, b), '\n')
#print('Ac: \n', np.matmul(A, c))

legend = []

n = int(1 / 4 / 0.0025)
Z = np.random.randn(n)
legend.append('Gaussian')

plt.step(sorted(Z), np.arange(1, n+1) / float(n))
plt.xlabel('Observations')
plt.ylabel('Probability')
plt.xlim([-3, 3])

for i in range(4):
    curr = np.sum(np.sign(np.random.randn(n, 8 ** i)) * np.sqrt(1./(8 ** i)), axis=1)
    plt.step(sorted(curr), np.arange(1, n+1) / float(n))
    legend.append(8 ** i)

plt.legend(legend)
plt.show()
