"""
Functions for the activation functions and the neural network basis function and derivatives.

Functions:
    - tanh: Hyperbolic tangent activation function
    - sin: Sine activation function
    - phi: Neural network basis function
    - phi_dx: Neural network basis function first derivative
    - phi_dxx: Neural network basis function second derivative
"""

import jax.numpy as jnp

#Activation functions
def tanh(x):
    return jnp.tanh(x)

def sin(x):
    return jnp.sin(x)


def phi(x, activation, weight, bias, mu, sd):
    x = activation(jnp.dot(x, weight) + bias)
    return x

def phi_dx(x, activation_dx, weight, bias, mu, sd):
    x = jnp.dot(activation_dx(jnp.dot(x, weight) + bias), weight)
    return x

def phi_dxx(x, activation_dxx, weight, bias, mu, sd):
    x = jnp.dot(activation_dxx(jnp.dot(x, weight) + bias), jnp.square(weight))
    return x
