from utils import calc_normalized_l1_loss
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import time


def solve_system(M, f_x):
    start_time = time.time()

    M_csc = sp.csc_matrix(M)  # Convert to CSC format for efficient solving
    LHS = M_csc.T @ M_csc
    RHS = M_csc.T @ f_x

    start_time = time.time()
    a = splinalg.spsolve(LHS, RHS)
    print("=" * 50)
    print(LHS @ a - RHS)
    print("=" * 50)
    end_time = time.time()

    elapsed_time = end_time - start_time

    loss = calc_normalized_l1_loss(a, M, f_x)

    print(f"Time taken for solver: {elapsed_time:.4f} seconds")
    return a, loss, elapsed_time


def solve_system_with_BCs(M_csc, B, lmda, f, g):

    start_time = time.time()

    # M_csc = sp.csc_matrix(M)
    B_csc = sp.csc_matrix(B)
    LHS = M_csc.T @ M_csc + lmda * B_csc.T @ B_csc
    RHS = M_csc.T @ f + lmda * B_csc.T @ g

    start_time = time.time()
    a = splinalg.spsolve(LHS, RHS)
    end_time = time.time()

    elapsed_time = end_time - start_time

    lhs_condition = np.linalg.cond(LHS.toarray())

    print(f"Time taken for solver: {elapsed_time:.4f} seconds")
    return a, elapsed_time, lhs_condition