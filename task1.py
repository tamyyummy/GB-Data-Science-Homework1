#Задание 1
import numpy as np

A = np.array([
    [1, 2, 7]
])
B = np.array([
    [3],
    [4],
    [5]
])

print(A.dot(B))
print(B.dot(A))

#Операция не коммутативна