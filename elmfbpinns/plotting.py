"""
Various plotting functions for window and partition of unity functions. 
Also includes plotting functions for the solution of the PDE.

Functions:
    plot_window_hat - Plots the window_hat function for each window.
    plot_window_hat_dx - Plots the derivative of the window_hat function for each window.
    plot_window_hat_dxx - Plots the second derivative of the window_hat function for each window.
    plot_POU - Plots the partition of unity function for each window.
    plot_POU_dx - Plots the derivative of the partition of unity function for each window.
    plot_POU_dxx - Plots the second derivative of the partition of unity function for each window.
    plot_solution - Plots the solution of the PDE and the function f(x).
"""


import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from windows import window_hat, window_hat_dx, window_hat_dxx, POU, POU_dx, POU_dxx, initInterval_old


def plot_window_hat(J, xmin_global, xmax_global, width, num_points=100):
    x = jnp.linspace(xmin_global, xmax_global, num_points)
    xmins, xmaxs = initInterval_old(J, xmin_global, xmax_global, width)

    plt.figure(figsize=(8, 6))
    for j in range(J):
        y = [window_hat(x[i], xmins[j], xmaxs[j]) for i in range(num_points)]
        plt.plot(x, y, label=f"Window {j+1} [{xmins[j]:.2f}, {xmaxs[j]:.2f}]")

    plt.title("Plot of window_hat function")
    plt.xlabel("x")
    plt.ylabel("window_hat(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_window_hat_dx(J, xmin_global, xmax_global, sd=None, num_points=100):
    x = jnp.linspace(xmin_global, xmax_global, num_points)
    xmins, xmaxs = initInterval(J, xmin_global, xmax_global, sd=sd)

    plt.figure(figsize=(8, 6))
    for j in range(J):
        y = [window_hat_dx(x[i], xmins[j], xmaxs[j]) for i in range(num_points)]
        plt.plot(x, y, label=f"Window {j+1} [{xmins[j]:.2f}, {xmaxs[j]:.2f}]")

    plt.title("Plot of window_hat function")
    plt.xlabel("x")
    plt.ylabel("window_hat(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_window_hat_dxx(J, xmin_global, xmax_global, sd=None, num_points=100):
    x = jnp.linspace(xmin_global, xmax_global, num_points)
    xmins, xmaxs = initInterval(J, xmin_global, xmax_global, sd=sd)

    plt.figure(figsize=(8, 6))
    for j in range(J):
        y = [window_hat_dxx(x[i], xmins[j], xmaxs[j]) for i in range(num_points)]
        plt.plot(x, y, label=f"Window {j+1} [{xmins[j]:.2f}, {xmaxs[j]:.2f}]")

    plt.title("Plot of window_hat function")
    plt.xlabel("x")
    plt.ylabel("window_hat(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_POU(J, global_xmin, global_xmax):
    x_plot = jnp.linspace(global_xmin, global_xmax, 100)
    colors = ["b", "g", "r", "c", "m"]

    xmins, xmaxs, sd = initInterval(J, global_xmin, global_xmax)

    plt.figure(figsize=(10, 6))

    omega_j = jnp.zeros_like(x_plot)
    for j in range(J):
        omega_j = [POU(x, j, xmins, xmaxs, J) for x in x_plot]
        plt.plot(x_plot, omega_j, label=f"Window {j}", color=colors[j])

    plt.title("Window Functions")
    plt.xlabel("x")
    plt.ylabel("Window Function Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_POU_dx(J, global_xmin, global_xmax):

    x_plot = jnp.linspace(global_xmin, global_xmax, 100)
    colors = ["b", "g", "r", "c", "m"]
    xmins, xmaxs, sd = initInterval(J, global_xmin, global_xmax)
    plt.figure(figsize=(10, 6))

    omega_j = jnp.zeros_like(x_plot)
    for j in range(J):
        omega_j = [POU_dx(x, j, xmins, xmaxs, J) for x in x_plot]
        plt.plot(x_plot, omega_j, label=f"Window {j}", color=colors[j])

    plt.title("Window Functions")
    plt.xlabel("x")
    plt.ylabel("Window Function Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_POU_dxx(J, global_xmin, global_xmax):
    x_plot = jnp.linspace(global_xmin, global_xmax, 100)
    colors = ["b", "g", "r", "c", "m"]
    xmins, xmaxs, sd = initInterval(J, global_xmin, global_xmax)
    plt.figure(figsize=(10, 6))

    omega_j = jnp.zeros_like(x_plot)
    for j in range(J):
        omega_j = [POU_dxx(x, j, xmins, xmaxs, J) for x in x_plot]
        plt.plot(x_plot, omega_j, label=f"Window {j}", color=colors[j])

    plt.title("Window Functions")
    plt.xlabel("x")
    plt.ylabel("Window Function Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_solution(x, u, f_solution, title):

    plt.figure(figsize=(10, 6))
    plt.plot(x, u, label="Solution u", marker="o")
    plt.plot(x, f_solution, label="Function f(x)", marker="x")
    plt.title(f"{title}")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
