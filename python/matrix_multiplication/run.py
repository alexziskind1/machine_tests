

import numpy as np
import time

# Size of the square matrices
N = 16000

# Generate random matrices
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# Record start time
start_time = time.time()

# Perform matrix multiplication
C = np.dot(A, B)

# Record end time
elapsed_time = time.time() - start_time
print(f"Matrix multiplication of size {N}x{N} took {elapsed_time:.2f} seconds")
