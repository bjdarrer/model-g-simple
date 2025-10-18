import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Constants
X_0 = 15.5556
G_0 = 15.5556
Y_0 = 1.8643

# Spatial and temporal parameters
length = 10.0  # Spatial domain length
dx = 0.05       # Spatial step size
dt = 0.01      # Time step size
num_steps = 10  # Number of time steps
c1 = 0 # BJD added this on 18.10.2025

# Create spatial grid
x = np.arange(-length / 2, length / 2, dx, dtype=np.float32)
N = len(x)

# Initialize fields with boundary conditions
phi_G = tf.Variable(tf.zeros(N), dtype=tf.float32)
phi_X = tf.Variable(tf.zeros(N), dtype=tf.float32)
phi_Y = tf.Variable(tf.zeros(N), dtype=tf.float32)

# Laplacian operator using finite differences with Dirichlet boundary conditions
def laplacian(field):
    field_padded = tf.concat([[0.0], field[1:-1], [0.0]], axis=0)  # Apply zero Dirichlet BCs
    return (tf.roll(field_padded, shift=-1, axis=0) - 2 * field_padded + tf.roll(field_padded, shift=1, axis=0)) / dx**2

# Seed fluctuation chi(x, t)
def chi(x, t, T=10.0, C=18.0):
    x = tf.cast(x, tf.float32)
    t = tf.cast(t, tf.float32)
    return -tf.exp(-x**2 / 2) * tf.exp(-((t - T)**2) / C)

# Time evolution
for step in range(num_steps):
    t = step * dt

    # Compute reaction terms
    reaction_G = -phi_G + phi_X / 10
    reaction_X = phi_G - 30 * phi_X - 4060/9 + (phi_X + 140/9)**2 * (phi_Y + 261 / 140)
    reaction_Y = 29 * phi_X + 4060/9 - (phi_X + 140/9)**2 * (phi_Y + 261 / 140)

    # Compute diffusion terms
    diffusion_G = laplacian(phi_G)
    diffusion_X = laplacian(phi_X)
    diffusion_Y = 12 * laplacian(phi_Y)

    # Compute seed fluctuation
    fluctuation = chi(x, t) if t <= 20 else tf.zeros_like(x, dtype=tf.float32)

    # Update fields with boundary conditions
    phi_G.assign(tf.maximum(-G_0, phi_G + dt * (diffusion_G + reaction_G)))
    phi_X.assign(tf.maximum(-X_0, phi_X + dt * (diffusion_X + reaction_X + fluctuation)))
    phi_Y.assign(tf.maximum(-Y_0, phi_Y + dt * (diffusion_Y + reaction_Y)))

    # Apply zero Dirichlet boundary conditions without reassigning the variable type
    phi_G = phi_G.numpy()
    phi_X = phi_X.numpy()
    phi_Y = phi_Y.numpy()

    phi_G[0], phi_G[-1] = 0.0, 0.0
    phi_X[0], phi_X[-1] = 0.0, 0.0
    phi_Y[0], phi_Y[-1] = 0.0, 0.0

    # Convert back to TensorFlow variables
    phi_G = tf.Variable(phi_G)
    phi_X = tf.Variable(phi_X)
    phi_Y = tf.Variable(phi_Y)

    # Update fields with boundary conditions
    #phi_G.assign(tf.maximum(-G_0, phi_G + dt * (diffusion_G + reaction_G)))
    #phi_X.assign(tf.maximum(-X_0, phi_X + dt * (diffusion_X + reaction_X + fluctuation)))
    #phi_Y.assign(tf.maximum(-Y_0, phi_Y + dt * (diffusion_Y + reaction_Y)))
    #phi_G = tf.tensor_scatter_nd_update(phi_G, [[0], [N - 1]], [0.0, 0.0])
    #phi_X = tf.tensor_scatter_nd_update(phi_X, [[0], [N - 1]], [0.0, 0.0])
    #phi_Y = tf.tensor_scatter_nd_update(phi_Y, [[0], [N - 1]], [0.0, 0.0])

    # Visualization every 1000 steps
    if step % 1== 0:
        c1 = c1 + 1
        plt.figure(figsize=(10, 6))
        plt.plot(x, phi_G.numpy(), label=r'$\phi_G$')
        plt.plot(x, phi_X.numpy(), label=r'$\phi_X$')
        plt.plot(x, phi_Y.numpy(), label=r'$\phi_Y$')
        plt.xlabel('Position')
        plt.ylabel('Potential Deviations')
        plt.title(f'Time = {t:.2f}')
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig('/home/brendan/software/model_g/plots/model_g_test_1c_' + str(c1) + '.png')
