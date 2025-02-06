"""
Functions that build training matrix, including boundary conditions, solve the system, and 
build the resulting solution matrix to generate a solution.

Functions:
    generate_indices: Generates the indices for the non-zero entries in the pseudo-mass matrices according to the subdomain boundaries.
    vectorized_matrix_entry: Vectorized function that computes the entries of the training matrix.
    elmfbpinn: Main function that builds the training matrix, solves the system, and generates the solution matrix.
"""

import jax
import jax.numpy as jnp
import time
import jax.random as random
import scipy
import pandas as pd


from windows import initInterval
from solvers import solve_system_with_BCs
from problems import compute_M_entry_vmap, compute_u_value_vmap, compute_du_value, compute_u_value
from utils import calc_l1_loss
from plotting import plot_solution, plot_window_hat
from networks import phi, phi_dx, phi_dxx

# [1,64,56,128,32,1]

# C - output basis functions

def generate_indices(J, C, xmins, xmaxs, x):
    row_indices = []
    col_indices = []

    J_array = jnp.arange(J)
    C_array = jnp.arange(C)

    #find where the values should exist
    def condition_check(x, j,c):
        condition = jnp.logical_and(x >= xmins[j], x <= xmaxs[j])
        # condition = (jnp.all(condition,0))
        return jnp.where(
            condition,
            1,
            0,
            )
    
    #vmap over the x values, j values, and c values
    vmap_condition_check = jax.vmap(jax.vmap(jax.vmap(condition_check, in_axes=(None, None, 0)), in_axes=(None, 0, None)), in_axes=(0, None, None))
    result = vmap_condition_check(x, J_array, C_array)
    
    result = result.reshape((x.shape[0], J * C)) #reshape the result to match the shape of M_ode, #needs altered for higher dimensions

    row_indices, col_indices= jnp.nonzero(result) #finds where the values should exist
    
    return row_indices, col_indices




def vectorized_matrix_entry(rows, columns, x_batch, J, C, basis, basis_dx, basis_dxx, params_hidden, xmins, xmaxs, sigma, compute_entry_func,debug=False):
    def single_M_entry(idx, basis, basis_dx, basis_dxx):
        # Print idx, row, col, j, c, and x
        row = rows[idx]
        col = columns[idx]
        j = col // C  # Extract j from column index
        c = col % C  # Extract c from column index
        x = x_batch[row]  # (1,)
        basis_row = basis[row]  # (32,)
        basis_dx_row = basis_dx[row]  # (32,)
        basis_dxx_row = basis_dxx[row]  # (32,)
        #jax.debug.print("Processing idx: {idx}, row: {row}, col: {col}, j: {j}, c: {c}, x: {x}", idx=idx, row=row, col=col, j=j, c=c, x=x)
        # jax.debug.print("basis_row[c]: {basis_row}", basis_row=basis_row[c])
        # jax.debug.print("basis_row[c]: {basis_dx_row}", basis_dx_row=basis_dx_row[c])
        # jax.debug.print("basis_row[c]: {basis_dxx_row}", basis_dxx_row=basis_dxx_row[c])
        entry_value = compute_entry_func(x, j, J, c, basis_row[c], basis_dx_row[c], basis_dxx_row[c], params_hidden, xmins, xmaxs, sigma)
        #jax.debug.print("Entry value: {entry_value}", entry_value=entry_value)
        return entry_value

    values = jax.vmap(
        single_M_entry, 
        in_axes=(0, None, None, None)  # vmap over rows, passing the basis arrays
    )(jnp.arange(rows.shape[0]), basis, basis_dx, basis_dxx)
    
    if debug:
        jax.debug.print(f"Values shape: {values}", values=values.shape)
    return values.reshape(-1)


def elmfbpinn(
    RHS_func,
    u,
    n_train,
    n_test,
    J,
    ranges,
    xmin,
    xmax,
    width,
    sigma,
    title,
    params_hidden,
    lmda=1,
    plot_window=True,
    debug=False,
):
    if debug:
        jax.debug.print("hidden_weights: {params_hidden}", params_hidden=params_hidden[0][0])
    total_start_time = time.time()  # Start time for the entire function

    xmins, xmaxs = ranges[:, 0], ranges[:, 1]
    if debug:
        print(f"xmins: {xmins}")
        print(f"xmaxs: {xmaxs}")
    
    C = params_hidden[-1][0].shape[1]  # C is equal to the number of neurons in the final hidden layer
    if debug:
        print(f"Number of neurons C: {C}")

    if plot_window:
        plot_window_hat(J, xmin, xmax, width)

    x_train = jnp.linspace(xmin, xmax, n_train).reshape(-1, 1)
    if debug:
        print(f"x_train shape: {x_train.shape}")
        print(f"x_train range: {x_train[0]} to {x_train[-1]}")

    # Generate indices for non-zero entries
    rows, columns = generate_indices(J, C, xmins, xmaxs, x_train)
    if debug:
        print(f"rows shape: {rows.shape}, columns shape: {columns.shape}")
        print(f"First few rows indices: {rows[:5]}")
        print(f"First few columns indices: {columns[:5]}")

    print("Creating M_ode...")
    start_time = time.time()

    basis = phi(x_train, params_hidden, sigma)
    
    basis_dx = phi_dx(x_train, params_hidden, sigma)

    basis_dxx = phi_dxx(x_train, params_hidden, sigma)
    if debug:
        print(f"basis shape: {basis.shape}")
        print(f"basis_dx shape: {basis_dx.shape}")
        print(f"basis_dxx shape: {basis_dxx.shape}")

    M_values = vectorized_matrix_entry(rows, columns, x_train, J, C, basis, basis_dx, basis_dxx, params_hidden, xmins, xmaxs, sigma, compute_M_entry_vmap)

    # Scale the values

    M_values_scaled = M_values / jnp.max(jnp.abs(M_values))
    

    # Create the sparse matrix using row and column indices
    M_ode_sparse = scipy.sparse.csc_matrix((M_values_scaled, (rows, columns)), shape=(n_train, J * C))
    if debug:
        print(f"M_values shape: {M_values.shape}")
        print(f"Max of M_values before scaling: {jnp.max(M_values)}")
        print(f"Max of M_values after scaling: {jnp.max(M_values_scaled)}")
        print(f"M_ode_sparse shape: {M_ode_sparse.shape}")
        print(f"M_ode_sparse non-zero entries: {M_ode_sparse.nnz}")
    
    M_dense = M_ode_sparse.toarray()
    # M_dense_df = pd.DataFrame(M_dense)
    # M_dense_df.to_csv("M_ode.csv")
    
    print(f"M_ode created in {time.time() - start_time:.2f} seconds.")

    # Exact solution
    exact_solution = RHS_func(x_train) / jnp.max(jnp.abs(M_values)).reshape(-1)
    print("Exact solution scaled.")

    # Initialize and compute B_train
    print("Creating B_train...")
    B_train = jnp.zeros((2, J * C))

    def single_B_train_entry(j, c):
        u_val = compute_u_value(
            x_train, 0, j, J, c, basis[0][c], basis_dx[0][c], basis_dxx[0][c], params_hidden, xmins, xmaxs, sigma
        )
        du_val = compute_du_value(
            x_train, 0, j, J, c, basis[0][c], basis_dx[0][c], basis_dxx[0][c], params_hidden, xmins, xmaxs, sigma
        )
        return u_val, du_val

    start_time = time.time()
    
    vmap_B_train_entry = jax.vmap(
        jax.vmap(single_B_train_entry, in_axes=(None, 0)), in_axes=(0, None)
    )
    
    u_vals, du_vals = vmap_B_train_entry(jnp.arange(J), jnp.arange(C))
    if debug:
        print(f"u_vals shape: {u_vals.shape}")
        print(f"du_vals shape: {du_vals.shape}")

    B_train = B_train.at[0].set(u_vals.reshape(J * C))
    B_train = B_train.at[1].set(du_vals.reshape(J * C))
    print(f"B_train created in {time.time() - start_time:.2f} seconds.")

    # Scaling B_train
    if debug:
        print(f"B_train shape: {B_train.shape}")
        print(f"B_train max values before scaling: {jnp.max(B_train)}")
    BD = jnp.diag(1.0 / jnp.max(jnp.abs(B_train), axis=1))
    B_ode_scaled = BD @ B_train

    # B_dense_df = pd.DataFrame(B_ode_scaled)
    # B_dense_df.to_csv("B_ode.csv")
    print("B_train scaled")

    # Boundary conditions
    g_train = jnp.zeros(2)
    g_train = BD @ g_train.at[0].set(1)

    # Solve the system with boundary conditions
    start_time = time.time()
    a, elapsed_time, lhs_condition = solve_system_with_BCs(
        M_ode_sparse, B_ode_scaled, lmda, exact_solution, g_train
    )
    print(f"a calculated in {time.time() - start_time:.2f} seconds.")
    if debug:
        print(f"a shape: {a.shape}")
        print(f"First few values of a: {a[:5]}")

    x_test = jnp.linspace(xmin, xmax, n_test).reshape(-1, 1)
    
    basis_test = phi(x_test, params_hidden, sigma)
    basis_dx_test = phi_dx(x_test, params_hidden, sigma)
    basis_dxx_test = phi_dxx(x_test, params_hidden, sigma)
    
    if debug:
        print(f"basis_test shape: {basis_test.shape}")
        print(f"basis_dx_test shape: {basis_dx_test.shape}")
        print(f"basis_dxx_test shape: {basis_dxx_test.shape}")

    print("Creating M_sol...")
    start_time = time.time()
    
    rows, columns = generate_indices(J, C, xmins, xmaxs, x_test)
    M_sol = vectorized_matrix_entry(rows, columns, x_test, J, C, basis_test, basis_dx_test, basis_dxx_test, params_hidden, xmins, xmaxs, sigma, compute_u_value_vmap)
    
    M_sol_sparse = scipy.sparse.csc_matrix((M_sol, (rows, columns)), shape=(x_test.shape[0], J * C))
    M_sol_dense = M_sol_sparse.toarray()
    # M_sol_dense_df = pd.DataFrame(M_sol_dense)
    # M_sol_dense_df.to_csv("M_sol.csv")
    
    print(f"M_sol created in {time.time() - start_time:.2f} seconds.")

    u_test = M_sol_sparse @ a
    if debug:
        print(f"u_test shape: {u_test.shape}")
        print(f"First few values of u_test: {u_test[:5]}")

    u_exact = u(x_test)
    if debug:
        print(f"First few values of u_exact: {u_exact[:5]}")

    # Plot the solution and print results.
    test_loss = calc_l1_loss(u_test, u_exact)
    print(f"Test Loss Value: {test_loss:.2e}")

    plot_solution(x_test, u_test, u_exact, title)

    print(f"Condition number of M_ode_sparse: {jnp.linalg.cond(M_ode_sparse.toarray()):.2e}")
    print(f"Condition number of M_sol_sparse: {jnp.linalg.cond(M_sol_sparse.toarray()):.2e}")
    print(f"Condition number of LHS: {lhs_condition:.2e}")

    loss = [test_loss]
    u = [u_test]
    Ms = [M_sol, M_ode_sparse, M_sol_sparse]
    B = [B_train]
    f = [exact_solution]
    x = [x_test]

    total_elapsed_time = time.time() - total_start_time  # Total time taken
    print(f"Total time taken: {total_elapsed_time:.2f} seconds.")

    return Ms, B, a, u, loss, x, f, lhs_condition, xmins, xmaxs, total_elapsed_time, rows, columns

