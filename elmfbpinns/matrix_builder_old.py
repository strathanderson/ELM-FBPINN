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


from windows import initInterval_old
from solvers import least_squares_solver
from problems_old import compute_M_entry_vmap_old, compute_u_value_vmap_old, compute_du_value_old, compute_u_value_old
from utils import calc_l1_loss
from plotting import plot_solution, plot_window_hat

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

# def generate_indices(C, xmins, xmaxs, x):
#     """
#     Generates row and column indices and col_ptrs for a CSC matrix format, based on which points in x fall
#     within the ranges defined by xmins and xmaxs for each subdomain (j, c).
#     """
#     # Expand x, xmins, and xmaxs to perform the full batch condition check
#     x_expanded = jnp.expand_dims(x, 1)  # (n, 1), for broadcasting
#     xmins_expanded = jnp.expand_dims(xmins, 0)  # (1, J), for broadcasting
#     xmaxs_expanded = jnp.expand_dims(xmaxs, 0)  # (1, J), for broadcasting

#     # Condition check across the full batch
#     inside = (x_expanded >= xmins_expanded) & (x_expanded <= xmaxs_expanded)  # (n, J) - Return True if x is within J subdomain
#     inside = jnp.repeat(inside, C, axis=1)  # Repeat J times for each C subdomain to get (n, J * C) - Copy the boolean results for each C within each J

#     # Find indices where the condition is True
#     row_indices, col_indices = jnp.nonzero(inside)

#     # Stack indices for output
#     indices = jnp.stack([row_indices, col_indices], axis=1)

#     return row_indices, col_indices

def vectorized_matrix_entry_old(rows, columns, x_batch, J, C, weights, biases, xmins, xmaxs, sigma, compute_entry_func):
    def single_M_entry(idx):
        row = rows[idx]
        col = columns[idx]
        j = col // C  # Extract j from column index
        c = col % C   # Extract c from column index
        x = x_batch[row]
        #jax.debug.print("Processing idx: {idx}, row: {row}, col: {col}, j: {j}, c: {c}, x: {x}", idx=idx, row=row, col=col, j=j, c=c, x=x)
        entry_value = compute_entry_func(x, j, J, c, weights, biases, xmins, xmaxs, sigma)
        #jax.debug.print("Entry value: {entry_value}", entry_value=entry_value)
        return entry_value

    values = jax.vmap(single_M_entry)(jnp.arange(rows.shape[0]))
    jax.debug.print("Values shape: {values}", values=values.shape)
    return values

def elmfbpinn_old(
    RHS_func,
    u,
    n_train,
    n_test,
    J,
    C,
    R,
    xmin,
    xmax,
    width,
    sigma,
    title,
    # key,
    # w_key,
    # b_key,
    weights=None,
    biases=None,
    lmda=1,
    plot_window=True,
    debug=False,
):
    total_start_time = time.time()  # Start time for the entire function
    #print(f"Total start time: {total_start_time}")

    if weights is None:
        key = random.PRNGKey(0)
        w_key, b_key = random.split(key)
        weights = random.uniform(w_key, shape=(C,), minval=-R, maxval=R)  # shape=(C,) (32,)
        biases = random.uniform(b_key, shape=(C,), minval=-R, maxval=R)
    
    print("====================================")
    print(f"Weights shapes: {weights.shape}")
    print(f"Weights: {weights}")
    print(f"Biases shapes: {biases.shape}")
    print(f"Biases: {biases}")

    xmins, xmaxs = initInterval_old(J, xmin, xmax, width=width, verbose=False)
    print("====================================")
    print(f"xmins: {xmins}")
    print(f"xmaxs: {xmaxs}")

    print("====================================")
    print(f"Number of neurons C: {C}")

    if plot_window:
        plot_window_hat(J, xmin, xmax, width)

    x_train = jnp.linspace(xmin, xmax, n_train)
    print("====================================")
    print(f"x_train shape: {x_train.shape}")
    print(f"First few x_train values: {x_train[:5]}")

    # Generate indices for non-zero entries
    rows, columns = generate_indices(J, C, xmins, xmaxs, x_train)
    print("====================================")
    print(f"rows shape: {rows.shape}, columns shape: {columns.shape}")
    print(f"First few rows indices: {rows[:5]}")
    print(f"First few columns indices: {columns[:5]}")

    print("Creating M_ode...")
    start_time = time.time()
    M_values = vectorized_matrix_entry_old(rows, columns, x_train, J, C, weights, biases, xmins, xmaxs, sigma, compute_M_entry_vmap_old)
    print("====================================")
    print(f"M_values shape: {M_values.shape}")
    print(f"Max of M_values before scaling: {jnp.max(M_values)}")
    print(f"First few values of M_values: {M_values[:5]}")

    # Scale the values
    M_values_scaled = M_values / jnp.max(jnp.abs(M_values))
    print("====================================")
    print(f"M_values_scaled shape: {M_values_scaled.shape}")
    print(f"Max of M_values_scaled: {jnp.max(M_values_scaled)}")
    print(f"First few values of M_values_scaled: {M_values_scaled[:5]}")

    # Create the sparse matrix using row and column indices
    M_ode_sparse = scipy.sparse.csc_matrix((M_values_scaled, (rows, columns)), shape=(n_train, J * C))
    print("====================================")
    print(f"M_ode_sparse shape: {M_ode_sparse.shape}")
    print(f"M_ode_sparse non-zero entries: {M_ode_sparse.nnz}")
    print(f"M_ode created in {time.time() - start_time:.2f} seconds.")

    # Exact solution
    exact_solution = RHS_func(x_train) / jnp.max(jnp.abs(M_values))
    print("====================================")
    print(f"exact_solution shape: {exact_solution.shape}")
    print(f"First few values of exact_solution: {exact_solution[:5]}")
    print("Exact_solution scaled.")

    # Initialize and compute B_train
    print("====================================")
    print("Creating B_train...")
    B_train = jnp.zeros((2, J * C))

    def single_B_train_entry(j, c):
        u_val = compute_u_value_old(
            x_train, 0, j, J, c, weights, biases, xmins, xmaxs, sigma
        )
        du_val = compute_du_value_old(
            x_train, 0, j, J, c, weights, biases, xmins, xmaxs, sigma
        )

        return u_val, du_val
    
    

    start_time = time.time()
    vmap_B_train_entry = jax.vmap(
        jax.vmap(single_B_train_entry, in_axes=(None, 0)), in_axes=(0, None)
    )
    u_vals, du_vals = vmap_B_train_entry(jnp.arange(J), jnp.arange(C))
    print(f"u_vals shape: {u_vals.shape}")
    print(f"first few u_vals: {u_vals[:5]}")
    print(f"du_vals shape: {du_vals.shape}")
    print(f"first few du_vals: {du_vals[:5]}")

    B_train = B_train.at[0].set(u_vals.reshape(J * C))
    B_train = B_train.at[1].set(du_vals.reshape(J * C))
    print(f"B_train created in {time.time() - start_time:.2f} seconds.")

    # Scaling B_train
    print("====================================")
    print(f"B_train shape: {B_train.shape}")
    print(f"B_train max value before scaling: {jnp.max(B_train)}")
    print(f"First few values of B_train: {B_train[:5]}")

    BD = jnp.diag(1.0 / jnp.max(jnp.abs(B_train), axis=1))
    B_ode_scaled = BD @ B_train

    print("====================================")
    print("B_train scaled")
    print(f"B_train scaled shape: {B_ode_scaled.shape}")
    print(f"B_train scaled max value: {jnp.max(B_ode_scaled)}")
    print(f"First few values of B_train scaled: {B_ode_scaled[:5]}")

    # Boundary conditions
    g_train = jnp.zeros(2)
    g_train = BD @ g_train.at[0].set(1) #Fix this later
    print("====================================")
    print("g_train created")
    print(f"g_train values: {g_train}")

    # Solve the system with boundary conditions
    start_time = time.time()
    a, elapsed_time, lhs_condition = least_squares_solver(
        M_ode_sparse, B_ode_scaled, lmda, exact_solution, g_train
    )
    print("====================================")
    print(f"a calculated in {time.time() - start_time:.2f} seconds.")
    print(f"a shape: {a.shape}")
    print(f"First few values of a: {a[:5]}")
    print(f"Final 5 values of a: {a[-5:]}")

    x_test = jnp.linspace(xmin, xmax, n_test)
    print("====================================")
    print("Creating M_sol...")
    start_time = time.time()
    rows, columns = generate_indices(J, C, xmins, xmaxs, x_test)

    M_sol = vectorized_matrix_entry_old(rows, columns, x_test, J, C, weights, biases, xmins, xmaxs, sigma, compute_u_value_vmap_old)
    M_sol_sparse = scipy.sparse.csc_matrix((M_sol, (rows, columns)), shape=(x_test.shape[0], J * C))
    print(f"M_sol shape: {M_sol.shape}")
    print(f"M_sol values created. Max value: {jnp.max(M_sol)}")
    print(f"M_sol created in {time.time() - start_time:.2f} seconds.")

    u_test = M_sol_sparse @ a

    if debug:
        print(f"u_test shape: {u_test.shape}")
        
        
    u_exact = u(x_test)

    u_exact = u(x_test)
    # if debug:
    print(f"First few values of u_exact: {u_exact[:5]}")
    print(f"First few values of u_test: {u_test[:5]}")

    # Plot the solution and print results.
    test_loss = calc_l1_loss(u_test, u_exact)
    print(f"Test Loss Value: {test_loss:.2e}")

    plot_solution(x_test, u_test, u_exact, title)

    print(f"Condition number of M_ode_sparse: {jnp.linalg.cond(M_ode_sparse.toarray()):.2e}")
    print(f"Condition number of M_sol_sparse: {jnp.linalg.cond(M_sol_sparse.toarray()):.2e}")
    print(f"Condition Number of LHS: {lhs_condition:.2e}")

    loss = [test_loss]
    u = [u_test, u_exact]
    Ms = [M_sol, M_ode_sparse, M_sol_sparse]
    B = [B_train]
    f = [exact_solution]
    x = [x_test]
    lsq = [M_ode_sparse, B_ode_scaled, lmda, exact_solution, g_train]

    total_elapsed_time = time.time() - total_start_time  # Total time taken
    print(f"Total time taken: {total_elapsed_time:.2f} seconds.")

    return lsq, Ms, B, a, u, loss, x, f, lhs_condition, xmins, xmaxs, total_elapsed_time, rows, columns


