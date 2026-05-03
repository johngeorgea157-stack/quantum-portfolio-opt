import numpy as np
import time
import math


def simulated_annealing_qubo(
    Q, k, T_start=1000.0, cooling_rate=0.99, max_iter=10000, seed=None
):
    """
    Simulated Annealing heuristic for the QUBO formulation of the portfolio problem.
    Maintains exactly k selected assets at all times to satisfy the constraint naturally.

    Args:
        Q: QUBO matrix (numpy array)
        k: Int, number of stocks to pick
        T_start: Float, initial high temperature
        cooling_rate: Float, multiplier to cool temperature (e.g. 0.99)
        max_iter: Int, max iterations
        seed: Int, optional random seed for reproducibility

    Returns:
        best_x: numpy array (binary vector)
        best_obj: float
        exec_time: float
    """
    if seed is not None:
        np.random.seed(seed)

    start_time = time.time()
    n = len(Q)

    # Random initial valid portfolio (exactly k ones)
    x_current = np.zeros(n, dtype=int)
    random_indices = np.random.choice(n, k, replace=False)
    x_current[random_indices] = 1

    current_obj = float(x_current @ Q @ x_current)

    best_x = x_current.copy()
    best_obj = current_obj

    T = T_start

    for _ in range(max_iter):
        # Stop if temperature is virtually 0
        if T < 1e-8:
            break

        # Generate a neighbor: flip one 1 to 0, and one 0 to 1
        x_neighbor = x_current.copy()

        # Find indices of current 1s and 0s
        ones_idx = np.where(x_neighbor == 1)[0]
        zeros_idx = np.where(x_neighbor == 0)[0]

        if len(ones_idx) > 0 and len(zeros_idx) > 0:
            # Pick a random 1 to flip to 0
            flip_to_zero = np.random.choice(ones_idx)
            # Pick a random 0 to flip to 1
            flip_to_one = np.random.choice(zeros_idx)

            x_neighbor[flip_to_zero] = 0
            x_neighbor[flip_to_one] = 1

        neighbor_obj = float(x_neighbor @ Q @ x_neighbor)

        # Delta E (change in energy)
        delta = neighbor_obj - current_obj

        # If neighbor is better (delta < 0), accept it!
        # If neighbor is worse, accept with probability e^(-delta / T)
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            x_current = x_neighbor.copy()
            current_obj = neighbor_obj

            # Keep track of absolute best found so far
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = x_current.copy()

        # Cool down the temperature
        T *= cooling_rate

    exec_time = time.time() - start_time

    return best_x, best_obj, exec_time


if __name__ == "__main__":
    # Simple test case
    Q_test = np.array([[-10, 2, 2, 5], [2, -8, 3, 1], [2, 3, -6, 4], [5, 1, 4, -12]])

    print("Testing Simulated Annealing...")
    best_x, best_obj, t = simulated_annealing_qubo(Q_test, k=2)
    print(f"Best X: {best_x}")
    print(f"Best QUBO Score: {best_obj}")
    print(f"Time: {t:.4f}s")
