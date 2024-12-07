import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt


def generate_sparse_matrix(rows, cols, density):
    np.random.seed(42)
    sparse_matrix = csr_matrix(
        np.random.choice([0, 1], size=(rows, cols), p=[1 - density, density]) *
        np.random.randint(1, 10, size=(rows, cols))
    )
    return sparse_matrix

def parallel_multiply(args):
    """Perform multiplication for a subset of rows in the sparse matrix."""
    row_indices, sparse_matrix, vector = args
    result = []
    for row_index in row_indices:
        row = sparse_matrix.getrow(row_index).toarray()
        result.append(np.dot(row, vector))
    return result

if __name__ == "__main__":

    rows, cols = 10000, 10000
    density = 0.0005
    sparse_matrix = generate_sparse_matrix(rows, cols, density)
    vector = np.random.randint(1, 10, size=cols)


    print("Starting sequential execution...")
    start_time = time.time()
    sequential_result = sparse_matrix @ vector
    sequential_time = time.time() - start_time
    print(f"Sequential Execution Time: {sequential_time:.4f}s")


    print("Starting parallel execution...")
    num_cores = cpu_count()
    row_chunks = np.array_split(range(rows), num_cores)

    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        parallel_results = pool.map(parallel_multiply, [(chunk, sparse_matrix, vector) for chunk in row_chunks])
    parallel_result = np.concatenate(parallel_results)
    parallel_time = time.time() - start_time
    print(f"Parallel Execution Time: {parallel_time:.4f}s")


    speedup = sequential_time / parallel_time
    efficiency = speedup / num_cores
    print(f"Speedup: {speedup:.2f}")
    print(f"Efficiency: {efficiency:.2f}")

    matrix_sizes = [1000, 5000, 10000, 20000]
    sequential_times = []
    parallel_times = []

    for size in matrix_sizes:
        print(f"Testing matrix size: {size}")
        sparse_matrix = generate_sparse_matrix(size, size, density)
        vector = np.random.randint(1, 10, size=size)

        start_time = time.time()
        sequential_result = sparse_matrix @ vector
        sequential_times.append(time.time() - start_time)

        row_chunks = np.array_split(range(size), num_cores)
        start_time = time.time()
        with Pool(processes=num_cores) as pool:
            pool.map(parallel_multiply, [(chunk, sparse_matrix, vector) for chunk in row_chunks])
        parallel_times.append(time.time() - start_time)

    plt.plot(matrix_sizes, sequential_times, label="Sequential Execution")
    plt.plot(matrix_sizes, parallel_times, label="Parallel Execution")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Performance Analysis: Sequential vs Parallel")
    plt.legend()
    plt.grid()
    plt.show()
