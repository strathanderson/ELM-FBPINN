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