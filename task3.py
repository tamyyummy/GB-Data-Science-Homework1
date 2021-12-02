import numpy as np
import time

def exec_and_measure(fun, m1, m2):
    start_time = time.time()
    result = fun(m1, m2)
    print(result)
    print("--- %.2f ms ---\n" % ((time.time() - start_time) * 1000))
    return result

X1 = 2
X2 = 3
num_features = 3
np.random.seed(42)
X1_mat = np.random.rand(X1, num_features)
X2_mat = np.random.rand(X2, num_features)
#print(X1_mat)
#print(X2_mat)
def compute_distances_two_loops(X1_mat, X2_mat):
    result = np.empty([X1_mat.shape[0], X2_mat.shape[0]])
    for row1 in range(0, X1_mat.shape[0]):
        for row2 in range(0, X2_mat.shape[0]):
            result[row1][row2] = np.linalg.norm(X1_mat[row1] - X2_mat[row2])
    return result

distances = exec_and_measure(compute_distances_two_loops, X1_mat, X2_mat)

def compute_distances_one_loop(X1_mat, X2_mat):
    result = np.empty([X1_mat.shape[0], X2_mat.shape[0]])
    for row in range(0, X1_mat.shape[0]):
        result[row] = np.linalg.norm(X2_mat - X1_mat[row], axis=1)
    return result

distances = exec_and_measure(compute_distances_one_loop, X1_mat, X2_mat)

def compute_distances_no_loops(X1_mat, X2_mat):
    a_ajusted = np.repeat(X1_mat, X2_mat.shape[0], axis=0) #выравнивание размерности матриц
    b_ajusted = np.tile(X2_mat, (X1_mat.shape[0], 1)) #выравнивание размерности матриц
    return np.linalg.norm(b_ajusted-a_ajusted, axis=1).reshape(X1_mat.shape[0], X2_mat.shape[0])

distances = exec_and_measure(compute_distances_no_loops, X1_mat, X2_mat)