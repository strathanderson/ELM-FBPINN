"""
Functions that use sparse solvers to solve the linear systems of equations
"""

from utils import calc_normalized_l1_loss, calc_l1_loss
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg #jax
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

    u_train = M_csc @ a
    loss = calc_normalized_l1_loss(u_train, f_x)

    print(f"Training loss: {loss:.2e}")

    print(f"Time taken for solver: {elapsed_time:.4f} seconds")
    return a, loss, elapsed_time


def least_squares_solver(M_csc, B, lmda, f, g):

    start_time = time.time()
    
    # M_csc = sp.csc_matrix(M)
    B_csc = sp.csc_matrix(B)

    LHS = M_csc.T @ M_csc + lmda * B_csc.T @ B_csc
    # print(f"LHS.shape: {LHS.shape}")
    # print(f"LHS: {LHS.toarray()}")
    # print(f"LHS condition number: {np.linalg.cond(LHS.toarray())}")
    RHS = M_csc.T @ f + lmda * B_csc.T @ g
    # print(f"RHS.shape: {RHS.shape}")
    # print(f"RHS: {RHS}")

    start_time = time.time()
    a = splinalg.spsolve(LHS, RHS)
    end_time = time.time()

    elapsed_time = end_time - start_time

    u_train = (M_csc @ a)

    training_loss = np.linalg.norm(u_train - f, ord=2) ** 2

    full_training_loss = np.linalg.norm(M_csc @ a - f, ord=2) ** 2 + lmda * np.linalg.norm(B_csc @ a - g, ord=2) ** 2

    print(f"Training loss: {training_loss:.2e}")
    print(f"Full training loss: {full_training_loss:.2e}")

    lhs_condition = np.linalg.cond(LHS.toarray())

    print(f"Time taken for solver: {elapsed_time:.4f} seconds")
    return a, elapsed_time, lhs_condition, training_loss, full_training_loss