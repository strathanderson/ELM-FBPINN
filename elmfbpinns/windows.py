"""
Functions used to define the subdomain intervals, window functions, and POU functions
"""


import jax.numpy as jnp
from utils import display_windows

def initInterval_old(J, xmin, xmax, width=1.9, verbose=False):
    sd = (xmax - xmin) / J
    xc = jnp.linspace(xmin, xmax, J)
    xmins = xc - width * sd
    xmaxs = xc + width * sd
    if verbose:
        display_windows(xmins, xmaxs)
    return xmins, xmaxs

def initInterval(nSubdomains, xmin, xmax, width=1.9, verbose=False):
    sd = (xmax - xmin) / nSubdomains
    xc = jnp.linspace(xmin, xmax, nSubdomains)
    xmins = xc - width*sd
    xmaxs = xc + width*sd
    if verbose:
        display_windows(xmins, xmaxs)
    return jnp.stack([xmins, xmaxs], axis=-1)

def norm(mu, sd, x):
    return (x-mu)/sd

def unnorm(mu, sd, x):
    return x*sd + mu

def cosine(xmin, xmax, x):
    mu, sd = (xmin+xmax)/2, (xmax-xmin)/2
    ws = ((1+jnp.cos(jnp.pi*(x-mu)/sd))/2)**2
    ws = jnp.heaviside(x-xmin,1)*jnp.heaviside(xmax-x,1)*ws
    w = jnp.prod(ws, axis=0, keepdims=True)
    return w

# Window functions
def window_hat(x, xmin, xmax):
    condition = jnp.logical_and(x >= xmin, x <= xmax)
    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0
    return jnp.where(condition, (1 + jnp.cos(jnp.pi * (x - mu) / sd)) ** 2, 0)

def window_hat_dx(x, xmin, xmax):
    condition = jnp.logical_and(x >= xmin, x <= xmax)
    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0
    return jnp.where(
        condition,
        -2
        * jnp.pi
        * (1 + jnp.cos(jnp.pi * (x - mu) / sd))
        * jnp.sin(jnp.pi * (x - mu) / sd)
        / sd,
        0,
    )

def window_hat_dxx(x, xmin, xmax):
    condition = jnp.logical_and(x >= xmin, x <= xmax)
    mu = (xmin + xmax) / 2.0
    sd = (xmax - xmin) / 2.0
    pi_term = jnp.pi * (x - mu) / sd

    term1 = 2 * jnp.pi**2 * jnp.sin(pi_term) ** 2 / sd**2
    term2 = 2 * jnp.pi**2 * jnp.cos(pi_term) * (1 + jnp.cos(pi_term)) / sd**2

    result = term1 - term2
    return jnp.where(condition, result, 0)

#POU functions

def POU(x, j, xmins, xmaxs, J):
    w_k = jnp.array([window_hat(x, xmins[k], xmaxs[k]) for k in range(J)])
    w_j = window_hat(x, xmins[j], xmaxs[j])

    return w_j / jnp.sum(w_k)

# POU_dx = jax.grad(POU)
# POU_dxx = jax.grad(POU_dx)


def POU_dx(x, j, xmins, xmaxs, J):
    w_k = jnp.array([window_hat(x, xmins[k], xmaxs[k]) for k in range(J)])
    w_j = window_hat(x, xmins[j], xmaxs[j])

    dwdx = jnp.array([window_hat_dx(x, xmins[k], xmaxs[k]) for k in range(J)])
    dwjdx = window_hat_dx(x, xmins[j], xmaxs[j])

    return (dwjdx * jnp.sum(w_k) - w_j * jnp.sum(dwdx)) / jnp.sum(w_k) ** 2


def POU_dxx(x, j, xmins, xmaxs, J):
    w_k = jnp.array([window_hat(x, xmins[k], xmaxs[k]) for k in range(J)])
    w_j = window_hat(x, xmins[j], xmaxs[j])

    dwdx = jnp.array([window_hat_dx(x, xmins[k], xmaxs[k]) for k in range(J)])
    dwjdx = window_hat_dx(x, xmins[j], xmaxs[j])

    dwdxx = jnp.array([window_hat_dxx(x, xmins[k], xmaxs[k]) for k in range(J)])
    dwjdxx = window_hat_dxx(x, xmins[j], xmaxs[j])

    sum_wk = jnp.sum(w_k)
    sum_dwdx = jnp.sum(dwdx)

    return (
        dwjdxx * sum_wk**2
        - 2 * dwjdx * sum_wk * sum_dwdx
        + 2 * w_j * sum_dwdx**2
        - w_j * jnp.sum(dwdxx) * sum_wk
    ) / sum_wk**3