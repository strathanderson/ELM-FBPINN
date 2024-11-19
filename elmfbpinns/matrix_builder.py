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


from windows import initInterval
from solvers import solve_system_with_BCs
from problems import compute_M_entry_vmap, compute_u_value_vmap, compute_du_value, compute_u_value
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

def vectorized_matrix_entry(rows, columns,x_batch, J, C, weights, biases, xmins, xmaxs, sigma,compute_entry_func):
    def single_M_entry(idx):
        row = rows[idx]
        col = columns[idx]
        j = col // C  # Extract j from column index
        c = col % C   # Extract c from column index
        x = x_batch[row]
        return compute_entry_func(x, j, J, c, weights, biases, xmins, xmaxs, sigma)

    values = jax.vmap(single_M_entry)(jnp.arange(rows.shape[0]))
    return values

#matrix builder
def elmfbpinn(
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
    weights=None,
    biases=None,
    lmda=1,
    plot_window=True,
):

    total_start_time = time.time()  # Start time for the entire function

    if weights is None:
        # Generate the weights and biases
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        weights = random.uniform(subkey, shape=(C,), minval=-R, maxval=R)

        key, subkey = random.split(key)
        biases = random.uniform(subkey, shape=(C,), minval=-R, maxval=R)

    xmins, xmaxs = initInterval(J, xmin, xmax, width=width)

    if plot_window:
        plot_window_hat(J, xmin, xmax, width)

    x_train = jnp.linspace(xmin, xmax, n_train)

    # Generate indices for non-zero entries
    rows, columns = generate_indices(J, C, xmins, xmaxs, x_train)
    # print(f"train indices: {indices}")
    # print(f"train rows: {rows}")
    # print(f"train columns: {columns}")

    print("Creating M_ode...")
    start_time = time.time()
    M_values = vectorized_matrix_entry(rows, columns,x_train, J, C, weights, biases, xmins, xmaxs, sigma,compute_M_entry_vmap)
    
    #scale the values
    M_values_scaled = M_values/jnp.max(jnp.abs(M_values))

    # Create the sparse matrix using row and column indices
    M_ode_sparse = scipy.sparse.csc_matrix((M_values_scaled, (rows, columns)), shape=(n_train, J * C))
    
    print(f"M_ode created in {time.time() - start_time:.2f} seconds.")

    # # Scale M_ode
    # MD = np.diag(1.0 / np.max(np.abs(M_ode_sparse.toarray()), axis=1))
    
    
    # M_ode_scaled = MD @ M_ode_sparse.toarray()
    # print("M_ode scaled.")

    # Changed this to a u instead of the f
    exact_solution = RHS_func(x_train)/jnp.max(jnp.abs(M_values))
    print("exact_solution scaled.")
    
    # Initialize and compute B_train
    print("Creating B_train...")
    B_train = jnp.zeros((2, J * C))

    def single_B_train_entry(j, c):
        u_val = compute_u_value(
            x_train, 0, j, J, c, weights, biases, xmins, xmaxs, sigma
        )
        du_val = compute_du_value(
            x_train, 0, j, J, c, weights, biases, xmins, xmaxs, sigma
        )
        return u_val, du_val

    start_time = time.time()
    vmap_B_train_entry = jax.vmap(
        jax.vmap(single_B_train_entry, in_axes=(None, 0)), in_axes=(0, None)
    )
    u_vals, du_vals = vmap_B_train_entry(jnp.arange(J), jnp.arange(C))

    B_train = B_train.at[0].set(u_vals.reshape(J * C))
    B_train = B_train.at[1].set(du_vals.reshape(J * C))
    print(f"B_train created in {time.time() - start_time:.2f} seconds.")

    # Scaling B_train
    BD = jnp.diag(1.0 / jnp.max(jnp.abs(B_train), axis=1))
    B_ode_scaled = BD @ B_train
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

    x_test = jnp.linspace(xmin, xmax, n_test)

    print("Creating M_sol...")
    start_time = time.time()
    
    rows, columns = generate_indices(J, C, xmins, xmaxs, x_test)
    
    M_sol = vectorized_matrix_entry(rows, columns,x_test, J, C, weights, biases, xmins, xmaxs, sigma,compute_u_value_vmap)
    M_sol_sparse = scipy.sparse.csc_matrix((M_sol, (rows, columns)), shape=(x_test.shape[0], J * C))
    print(f"M_sol created in {time.time() - start_time:.2f} seconds.")

    u_test = M_sol_sparse @ a
    u_exact = u(x_test)

    # Plot the solution and print results.
    test_loss = calc_l1_loss(u_test, u_exact)
    print(f"Test Loss Value: {test_loss:.2e}")

    plot_solution(x_test, u_test, u_exact, title)

    print(f"Condition number of M_ode_scaled: {jnp.linalg.cond(M_ode_sparse.toarray()):.2e}")
    print(f"Condition number of M_sol: {jnp.linalg.cond(M_sol_sparse.toarray()):.2e}")
    print(f"Condition Number of LHS: {lhs_condition:.2e}")

    loss = [test_loss]
    u = [u_test]
    Ms = [M_sol,M_ode_sparse,M_sol_sparse]
    B = [B_train]
    f = [exact_solution]
    x = [x_test]

    total_elapsed_time = time.time() - total_start_time  # Total time taken
    print(f"Total time taken: {total_elapsed_time:.2f} seconds.")

    return Ms, B, a, u, loss, x, f, lhs_condition, xmins, xmaxs, total_elapsed_time, rows, columns

