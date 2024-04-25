import CostFunction
from Node import *
import Node

import autograd.numpy as np

# node specific
node_id = 0
neighboring_nodes = np.array([1, 2]) # ID list

# parameters
epsilon = 0.001
min_accepted_divergence = 0.02
c = 0.00000001
MAX_ITER = 10000
IPM_cycle = 1
bb = 10
x0 = np.array([8, 800, 16, 7]) # initial point


# initialize the node as an object using the Node class
node = Node(node_id,
            x0,
            epsilon,
            c,
            min_accepted_divergence,
            neighboring_nodes,
            bb,
            CostFunction.CostFunction())


# outer loop
for cycle in range(IPM_cycle):

    # reset convergence flag for the next iterations
    CONVERGENCE_FLAG = False
    iter = 1

    # inner loop
    while not CONVERGENCE_FLAG:
        # transmission
        node.transmit_data()
        
        # receive message and update state
        node.receive_data()
        node.update_estimation(iter)

        # check if convergence is reached
        if node.has_converged() or iter >= MAX_ITER:
            CONVERGENCE_FLAG = True
            print(f"Reached convergence at iter:{iter}")
        else:
            iter += 1

    
    # update initial condition and steepness for the next cycle
    node.x0 = node.xi
    node.bb *= 1.5