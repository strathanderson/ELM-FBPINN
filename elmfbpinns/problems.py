"""

"""

import jax
import jax.numpy as jnp

from windows import POU, POU_dx, POU_dxx
from networks import phi, phi_dx, phi_dxx

#1D harmonic oscillator
def compute_M_entry_vmap(x, j, J, c, weights, biases, xmins, xmaxs, sigma):
    sigma, sigma_dx, sigma_dxx = sigma, jax.grad(sigma), jax.grad(jax.grad(sigma))
    if J == 1:
        partition_dxx = 0
        partition_dx = 0
        partition = 1
    else:
        partition = POU(x, j, xmins, xmaxs, J)
        partition_dx = POU_dx(x, j, xmins, xmaxs, J)
        partition_dxx = POU_dxx(x, j, xmins, xmaxs, J)


    mu = (xmins[j] + xmaxs[j]) / 2.0
    sd = (xmaxs[j] - xmins[j]) / 2.0

    basis = phi(x, sigma, weights[c], biases[c], mu, sd)
    basis_dx = phi_dx(x, sigma_dx, weights[c], biases[c], mu, sd)
    basis_dxx = phi_dxx(x, sigma_dxx, weights[c], biases[c], mu, sd)

    u_tt = partition_dxx * basis + 2 * partition_dx * basis_dx + partition * basis_dxx
    u_t = partition_dx * basis + partition * basis_dx
    u = partition * basis

    m = 1
    d = 2
    mu_param = 2 * d
    omega_0 = 80
    k = omega_0**2

    entry = m * u_tt + mu_param * u_t + (k + 1) * u

    return entry

def compute_u_value_vmap(x, j, J, c, weights, biases, xmins, xmaxs, sigma):
    sigma, sigma_dx, sigma_dxx = sigma, jax.grad(sigma), jax.grad(jax.grad(sigma))
    if J == 1:
        partition = 1
    else:
        partition = POU(x, j, xmins, xmaxs, J)

    mu = (xmins[j] + xmaxs[j]) / 2.0

    sd = (xmaxs[j] - xmins[j]) / 2.0

    basis = phi(x, sigma, weights[c], biases[c], mu, sd)

    u = partition * basis

    return u


def compute_du_value_vmap(x, j, J, c, weights, biases, xmins, xmaxs, sigma):
    sigma, sigma_dx, sigma_dxx = sigma, jax.grad(sigma), jax.grad(jax.grad(sigma))

    if J == 1:
        partition_dx = 0
        partition = 1
    else:
        partition = POU(x, j, xmins, xmaxs, J)
        partition_dx = POU_dx(x, j, xmins, xmaxs, J)

    mu = (xmins[j] + xmaxs[j]) / 2.0

    sd = (xmaxs[j] - xmins[j]) / 2.0

    basis = phi(x, sigma, weights[c], biases[c], mu, sd)
    basis_dx = phi_dx(x, sigma_dx, weights[c], biases[c], mu, sd)

    u_t = partition_dx * basis + partition * basis_dx

    return u_t

#functions for boundary condition
def compute_u_value(x, l, j, J, c, weights, biases, xmins, xmaxs, sigma):
    if J == 1:
        partition = 1
    else:
        partition = POU(x[l], j, xmins, xmaxs, J)

    mu = (xmins[j] + xmaxs[j]) / 2.0

    sd = (xmaxs[j] - xmins[j]) / 2.0

    basis = phi(x[l], sigma, weights[c], biases[c], mu, sd)

    u = partition * basis

    return u


def compute_du_value(x, l, j, J, c, weights, biases, xmins, xmaxs, sigma):

    if J == 1:
        partition_dx = 0
        partition = 1
    else:
        partition = POU(x[l], j, xmins, xmaxs, J)
        partition_dx = POU_dx(x[l], j, xmins, xmaxs, J)

    mu = (xmins[j] + xmaxs[j]) / 2.0

    sd = (xmaxs[j] - xmins[j]) / 2.0
    
    sigma_dx = jax.grad(sigma)

    basis = phi(x[l], sigma, weights[c], biases[c], mu, sd)
    basis_dx = phi_dx(x[l], sigma_dx, weights[c], biases[c], mu, sd)

    u_t = partition_dx * basis + partition * basis_dx

    return u_t

# Damped Harmonic Oscillator parameters
m = 1
delta = 2
mu = 2 * delta
omega_0 = 80
k = omega_0**2
omega = jnp.sqrt(omega_0**2 - delta**2)


# Damped harmonic oscillator solution and its derivatives
def harm_u_exact(t):
    phi = jnp.arctan(-delta / omega)
    A = 1 / (2 * jnp.cos(phi))
    return 2 * A * jnp.exp(-delta * t) * jnp.cos(phi + omega_0 * t)


def zero_RHS(t):
    return 0.0 * harm_u_exact(t)