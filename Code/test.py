import numpy as np

a = np.array([1, 1])
b = np.array([1, 3])

c = np.linalg.norm(b - a)

print(c)
