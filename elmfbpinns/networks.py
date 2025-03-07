"""
Functions for the activation functions and the neural network basis function and derivatives.

Functions:
    - tanh: Hyperbolic tangent activation function
    - sin: Sine activation function
    - phi: Neural network basis function
    - phi_dx: Neural network basis function first derivative
    - phi_dxx: Neural network basis function second derivative
"""
import jax
import jax.numpy as jnp
import jax.random as random
import optax

#Activation functions
def tanh(x):
    return jnp.tanh(x)

def sin(x):
    return jnp.sin(x)


def phi_old(x, activation, weight, bias, mu, sd): 
    x = activation(jnp.dot(x, weight) + bias)
    return x

def phi_dx_old(x, activation_dx, weight, bias, mu, sd):
    x = jnp.dot(activation_dx(jnp.dot(x, weight) + bias), weight)
    return x

def phi_dxx_old(x, activation_dxx, weight, bias, mu, sd):
    x = jnp.dot(activation_dxx(jnp.dot(x, weight) + bias), jnp.square(weight))
    return x

# def phi(x, activation, weights, biases, mu, sd):#xvalues #vmap over the weights and biases [this works with deep layers]
#     x_normalized = (x - mu) / sd
#     for weight, bias in zip(weights, biases):
#         x_normalized = activation(jnp.dot(x_normalized, weight) + bias)
#     return x_normalized

# phi_dx = jax.grad(phi)

def phi(x, params_hidden,sigma):
    x = x.reshape(-1,1)
    for weight, bias in params_hidden:
        x = sigma(jnp.dot(x, weight) + bias)
    #x=2*x**2 #For testing the jvps
    return x

def phi_dx(x,params_hidden, sigma):
    x = x.reshape(-1,1)
    u_fn = lambda x: phi(x,params_hidden,sigma)
    _, du = jax.jvp(u_fn, (x,), (jnp.ones_like(x),))
    return du

def phi_dxx(x,params_hidden, sigma):
    x = x.reshape(-1,1)
    du_fn = lambda x: phi_dx(x,params_hidden,sigma)
    _, ddu = jax.jvp(du_fn, (x,), (jnp.ones_like(x),))
    return ddu


def initWeightBiases(nNetworks, layer):
    params = []
    for i in range(len(layer) - 1):
        key = random.PRNGKey(i)
        w_key, b_key, key = random.split(key, 3)
        v = jnp.sqrt(2./(layer[i] + layer[i+1]))
        W = random.uniform(w_key, (nNetworks,layer[i+1], layer[i]), minval=-v, maxval=v)
        b = random.uniform(b_key, (nNetworks, layer[i+1]), minval=-v, maxval=v)
        params.append((W, b))
    return params

class FCN:
    def __init__(self, layers, learning_rate=1e-3):
        self.layers = layers
        self.learning_rate = learning_rate
        self.params_all = self.initWeightBiases(self.layers)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params_all)
    
    @staticmethod
    def initWeightBiases(layers, seed=42):
        key = random.PRNGKey(seed)
        params = []
        for i in range(len(layers) - 1):
            key, subkey = random.split(key)
            v = jnp.sqrt(2. / (layers[i] + layers[i + 1]))
            W = random.uniform(subkey, (layers[i + 1], layers[i]), minval=-v, maxval=v)
            b = random.uniform(subkey, (layers[i + 1],), minval=-v, maxval=v)
            params.append((W, b))
        return params

    @staticmethod
    def forward(params, x):
        for W, b in params[:-1]:
            x = jnp.tanh(jnp.dot(W, x) + b)
        W, b = params[-1]
        return jnp.dot(W, x) + b

    def model(self, params_all, x_batch):
        # Explicit loop instead of vmap
        u_pred = jnp.zeros((x_batch.shape[0],))  # Initialize the output array
        for i in range(x_batch.shape[0]):
            u_pred = u_pred.at[i].set(self.forward(params_all, x_batch[i]))  # Apply forward pass on each sample
        return u_pred

    def loss(self, params_all, x_batch, u_batch):
        u_pred = self.model(params_all, x_batch)
        return jnp.mean((u_batch - u_pred) ** 2)

    def train_model(self, x_train, u_train, num_epochs=500):
        @jax.jit
        def step(params, opt_state, x_batch, u_batch):
            loss_value, grads = jax.value_and_grad(self.loss)(params, x_batch, u_batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value
        
        for epoch in range(num_epochs):
            self.params_all, self.opt_state, loss_value = step(
                self.params_all, self.opt_state, x_train, u_train
            )
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss_value:.6f}")
    
    def predict(self, x_batch):
        return self.model(self.params_all, x_batch)

    def extract_weights_and_biases(self, params=None):
        if params is None:
            params = self.params_all  # Use the class's params_all if no argument is passed
            
        weights = [W for W, b in params]
        biases = [b for W, b in params]
        return weights, biases