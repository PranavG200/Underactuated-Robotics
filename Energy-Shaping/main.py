from matplotlib import rc
rc('animation', html='jshtml')

import numpy as np
from math import sin, cos, pi

import importlib
import cartpole
import cartpole_sim
importlib.reload(cartpole)
importlib.reload(cartpole_sim)
from cartpole import Cartpole
from cartpole_sim import simulate_cartpole

# Initial state
x0 = np.zeros(4)
x0[1] = pi/6
tf = 10

# Create a cartpole system with LQR controller
cartpole = Cartpole()

# Simulate and create animation
anim, fig = simulate_cartpole(cartpole, x0, tf, True)

anim
