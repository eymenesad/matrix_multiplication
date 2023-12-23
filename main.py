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

# WRITE YOUR CODE BELOW #
def convert_index(index_i, index_j, N, K):
    element = index_i * N + index_j
    block = element // (K * K)
    offset = element % (K * K)
    new1 = offset // K + K * (block // (N // K))
    new2 = offset % K + K * (block % (N // K))
    return new1, new2

offset_A = 0
offset_B = N * N
offset_C = 2 * N * N

A1 = np.zeros([N, N])
B1 = np.zeros([N, N])
# store A1 and B1 in a row major way
for i in range(N):
    for j in range(N):
        cs1.store(offset_A + i * N + j)
        cs1.store(offset_B + i * N + j)
        A1[i][j] = rnd_vals1[i][j]
        B1[i][j] = rnd_vals2[i][j]

A2 = np.zeros([N, N])
B2 = np.zeros([N, N])
for i in range(N):
    for j in range(N):
        new_i, new_j = convert_index(i, j, N, K)
        cs2.store(offset_A + new_i * N + new_j)
        cs2.store(offset_B + new_i * N + new_j)
        A2[new_i][new_j] = rnd_vals1[i][j]
        B2[new_i][new_j] = rnd_vals2[i][j]

# Matrix Multiplication Functions
def simple_multiplication():

    N = len(A1)
    C1 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                cs1.load(offset_A + i * N + k)      # load A[i][k]
                cs1.load(offset_B + k * N + j)      # load B[k][j]
                cs1.load(offset_C + i * N + j) 
                cs1.store(offset_C + i * N + j)     # store C[i][j]

                C1[i][j] += A1[i][k] * B1[k][j]
    
    # write now the block multiplication
    
    C2 = np.zeros((N, N)) 
    # block array
    for i in range(N):
        for j in range(N):
            for k in range(N):
                
                cs2.load(offset_A + i * N + k)      # load A[i][k]
                cs2.load(offset_B + k * N + j)      # load B[k][j]
                cs2.load(offset_C + i * N + j) 
                cs2.store(offset_C + i * N + j)     # store C[i][j]

                C2[i][j] += A2[i][k] * B2[k][j]
    
    return C1, C2


def recursive_multiplication():
    # Helper function for recursion
    
    def multiply_block(A, B, C, row_A, col_A, row_B, col_B, size, cs_inp):
        if size == 1:
            if cs_inp == 1:
                cs1.load(offset_A + (row_A) * N + col_A)
                cs1.load(offset_B + (row_B) * N + col_B)
                cs1.load(offset_C + (row_A) * N + col_B)
                cs1.store(offset_C + (row_A) * N + col_B)
            if cs_inp == 2:
                cs2.load(offset_A + (row_A) * N + col_A)
                cs2.load(offset_B + (row_B) * N + col_B)
                cs2.load(offset_C + (row_A) * N + col_B)
                cs2.store(offset_C + (row_A) * N + col_B)

            C[row_A][col_B] += A[row_A][col_A] * B[row_B][col_B]
        else:
            half = size // 2
            # Top Left
            multiply_block(A, B, C, row_A, col_A, row_B, col_B, half, cs_inp)
            multiply_block(A, B, C, row_A, col_A + half, row_B + half, col_B, half, cs_inp)
            # Top Right
            multiply_block(A, B, C, row_A, col_A, row_B, col_B + half, half, cs_inp)
            multiply_block(A, B, C, row_A, col_A + half, row_B + half, col_B + half, half, cs_inp)
            # Bottom Left
            multiply_block(A, B, C, row_A + half, col_A, row_B, col_B, half, cs_inp)
            multiply_block(A, B, C, row_A + half, col_A + half, row_B + half, col_B, half, cs_inp)
            # Bottom Right
            multiply_block(A, B, C, row_A + half, col_A, row_B, col_B + half, half, cs_inp)
            multiply_block(A, B, C, row_A + half, col_A + half, row_B + half, col_B + half, half, cs_inp)

    
   


    N = len(A1)
    R1 = np.zeros((N, N))
    multiply_block(A1, B1, R1, 0, 0, 0, 0, N, 1)

    
    N = len(A2)
    R2 = np.zeros((N, N))
    multiply_block(A2, B2, R2, 0 , 0, 0, 0, N, 2)


    return R1, R2

if algorithm == 'simple':
    #R1 ROW, R2 BLOCK
    S1, S2 = simple_multiplication()

    
elif algorithm == 'recursive':
    #B1 ROW, B2 BLOCK
    R1, R2 = recursive_multiplication()

    """ print(A1)
    print(B1)

    print("**************")

    print("a1.b1 with numpy:")
    print(np.dot(A1,B1))
    print("..............")
    print("r1")
    print(R1)
    print("---------------")
    
    print(A2)
    print(B2)
    print("*************")

    print("a2.b2 with numpy:")
    print(np.dot(A2,B2))
    print("..............")
    
    print("r2")
    print(R2)
    print("---------------") """

    
else:
    raise ValueError("Invalid algorithm choice")


# WRITE YOUR CODE ABOVE #

print('Row major array')
cs1.print_stats()


print('Block array')
cs2.print_stats()