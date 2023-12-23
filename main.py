import numpy as np

from cachesim import CacheSimulator, Cache, MainMemory
from argparse import ArgumentParser


def make_cache() -> CacheSimulator:
    mem = MainMemory()
    l3 = Cache("L3", 20480, 16, 64, "LRU")                           
    mem.load_to(l3)
    mem.store_from(l3)
    l2 = Cache("L2", 512, 8, 64, "LRU", store_to=l3, load_from=l3)  
    l1 = Cache("L1", 64, 8, 64, "LRU", store_to=l2, load_from=l2) 
    cs = CacheSimulator(l1, mem)
    return cs

sorted_list = []
parser = ArgumentParser()
parser.add_argument('-a', '--algorithm', type=str, choices=['simple', 'recursive'])
parser.add_argument('-N', '--N', type=int)
parser.add_argument('-K', '--K', type=int)
args = parser.parse_args()

algorithm, N, K = args.algorithm, args.N, args.K

cs1 = make_cache()
cs2 = make_cache()

rnd_vals1 = np.random.rand(N, N)
rnd_vals2 = np.random.rand(N, N)

# WRITE YOUR CODE BELOW #aaa
# Matrix Multiplication Functions
def simple_multiplication(matrix1, matrix2):
    N = len(matrix1)
    result = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result

def recursive_multiplication(matrix1, matrix2, size, block_size):
    # Helper function for recursion
    def multiply_block(A, B, result, row, col, size):
        if size == block_size:
            for i in range(block_size):
                for j in range(block_size):
                    sum = 0
                    for k in range(block_size):
                        sum += A[row + i][col + k] * B[col + k][row + j]
                    result[row + i][row + j] += sum
            return
        half = size // 2
        multiply_block(A, B, result, row, col, half)
        multiply_block(A, B, result, row, col + half, half)
        multiply_block(A, B, result, row + half, col, half)
        multiply_block(A, B, result, row + half, col + half, half)

    N = len(matrix1)
    result = np.zeros((N, N))
    multiply_block(matrix1, matrix2, result, 0, 0, N)
    return result

if algorithm == 'simple':
    result1 = simple_multiplication(rnd_vals1, rnd_vals2)
    result2 = simple_multiplication(rnd_vals1, rnd_vals2)
elif algorithm == 'recursive':
    result1 = recursive_multiplication(rnd_vals1, rnd_vals2, N, K)
    result2 = recursive_multiplication(rnd_vals1, rnd_vals2, N, K)
else:
    raise ValueError("Invalid algorithm choice")


# WRITE YOUR CODE ABOVE #

print('Row major array')
cs1.print_stats()


print('Block array')
cs2.print_stats()