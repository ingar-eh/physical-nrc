import numpy as np
import CostFunction
from Node import *

# functions

import random
import time
import numpy as np
import autograd.numpy as np
from autograd import jacobian


def __init__(self, node_id: int, x0: np.array, epsilon: float, c: float,
             minimum_accepted_divergence: float,
             adjacency_vector: np.array,
             bb: float,
             simulation_function_xtx_btx):


    self.node_id = node_id
    self.epsilon = epsilon
    self.xi = np.array(x0)
    self.bb = bb

    # list for storing evolution of signals
    self.all_calculated_xis = []
    self.evolution_costfun = []
    self.ratio_evol = []
    self.zi_evol = []

    self.simulation_function_xtx_btx = simulation_function_xtx_btx

    self.ff = simulation_function_xtx_btx.get_fn(self.xi, sea_cond=self.sea_condition, bb=self.bb)

    self.number_of_neighbors = np.sum(adjacency_vector)
    self.minimum_accepted_divergence = minimum_accepted_divergence
    self.adjacency_vector = adjacency_vector

    self.hi_old = np.eye(x0.size)
    self.hi = np.eye(x0.size)

    self.gi = np.zeros(x0.size).transpose()
    self.gi_old = np.zeros(x0.size).transpose()
    self.zi = np.eye(x0.size)  # self.hi_old
    self.yi = np.zeros(x0.size).transpose()  # self.gi_old

    self.c = c
    self.cI = c * np.eye(x0.size) # variable to be check in order to ensure robustness and large basin of attraction

    self.sigma_yi = np.zeros(x0.size).transpose()  # counter of the total mass-y sent
    self.sigma_zi = np.zeros((x0.size, x0.size))    # counter of the total mass-z sent, matrix x0.size X x0.size

    self.rho_yj = np.zeros((adjacency_vector.size, x0.size))  # counter of the total mass-y received from j, matrix_dim X xo.size
    self.rho_yj_old = np.zeros((adjacency_vector.size, x0.size))
    self.rho_zj = np.zeros((adjacency_vector.size, x0.size, x0.size))  # counter of the total mass-z received from j
    self.rho_zj_old = np.zeros((adjacency_vector.size, x0.size, x0.size))

    self.iter = 0

    self.msg_rel = True  # flag to simulate a packet loss

    self.is_ready_to_receive = False
    self.is_ready_to_update = False
    self.is_ready_to_transmit = True

    self.ratio = 0

#modify initial condition for the next cycle of the IPM
def initial_point_IPM(self, yi, zi):
    self.hi_old = self.simulation_function_xtx_btx.get_hessian_fn(self.xi, sea_cond=self.sea_condition, bb=self.bb)
    self.hi = self.simulation_function_xtx_btx.get_hessian_fn(self.xi, sea_cond=self.sea_condition, bb=self.bb)

    self.gi = np.subtract(np.matmul(self.hi, self.xi.transpose()),
                          self.simulation_function_xtx_btx.get_gradient_fn(self.xi, sea_cond=self.sea_condition, bb=self.bb))
    self.gi_old = np.subtract(np.matmul(self.hi, self.xi.transpose()),
                          self.simulation_function_xtx_btx.get_gradient_fn(self.xi, sea_cond=self.sea_condition, bb=self.bb))

    self.zi = zi  # start from the last value of previous cycle
    self.yi = yi.transpose()  # start from the last value of previous cycle
    return

def transmit_data(self):
    """This method update the yi and zi in each iteration and create a message including the new updated yi and
    zi. Finally the new message will be broadcast to all neighbors of this node. """
    print(f"Node ID: {self.node_id} -  transmitting data started!\n")

    self.msg_rel = True  # always reset the simulation flag, starting point is reliable

    self.yi = (1 / (self.number_of_neighbors + 1)) * self.yi
    self.zi = self.zi / (self.number_of_neighbors + 1)

    self.sigma_yi = self.sigma_yi + self.yi
    self.sigma_zi = self.sigma_zi + self.zi

    # simulate packet loss (not considered for the moment)
    if random.randrange(0, 9, 1) == 1:  # 1/10 chance to loss message
        self.msg_rel = False
        print("message LOST")

    '''
    if random.randrange(0, 9, 1)%2 == 0:  # 1/2 chance to loss message
        self.msg_rel = False
        print("message LOST")
    '''
    print(f"Node ID: {self.node_id} -  transmitting data ended!\n")
    return

def broadcast(self, message):
    """This method will broadcast the passed message to all neighbors of this node."""
    print(f"Node ID: {self.node_id} -  broadcasting started!\n")
    with self.lock:
        i = 0
        while i < len(self.all_nodes_message_buffers):
            if self.adjacency_vector[i] == 1:
                self.all_nodes_message_buffers[i].put(message)
            i += 1
    time.sleep(0.05)
    print(f"Node ID: {self.node_id} -  broadcasting ended!\n")
    return

def receive_data(self, message):
    """This method will handle the reception of data from neighbors. Using the yi and zi from the messages,
    the yi and zi would be updated by this method."""
    print(f"Node ID: {self.node_id} -  receiving data started!\n")
    self.j = message['node_id']

    # update old virtual mass received
    self.rho_zj_old[self.j] = self.rho_zj[self.j]
    self.rho_yj_old[self.j] = self.rho_yj[self.j]

    if message['msg_rel']:       # update virtual mass of neighbour
        self.rho_yj[self.j] = message['sigma_yi']
        self.rho_zj[self.j] = message['sigma_zi']

    # update estimate
    self.yi = self.yi + self.rho_yj[self.j] - self.rho_yj_old[self.j]
    self.zi = self.zi + self.rho_zj[self.j] - self.rho_zj_old[self.j]

    print(f"Node ID: {self.node_id} -  Receiving data ended!\n")
    return

def update_estimation(self, iter):
    """This method will calculate the next xi and also update the hi and gi by using the new xi. """
    print(f"Node ID: {self.node_id} -  Updating data started!\n")
    self.all_calculated_xis.append(self.xi)
    self.evolution_costfun.append(self.ff)
    while iter > len(self.evolution_costfun):
        self.all_calculated_xis.append(self.xi)
        self.evolution_costfun.append(self.ff)

    a = np.abs(np.linalg.eigvals(self.zi))

    # check condition on z
    if (np.abs(np.linalg.eigvals(self.zi)) < np.linalg.eigvals(self.cI)).all():
        self.zi = self.cI
    '''if (np.linalg.det(self.zi) <= self.c):
        self.zi = self.cI'''

    '''if iter >= 1000:
        self.epsilon = 0'''

    self.xi = (1 - self.epsilon) * self.xi + np.matmul((self.epsilon * np.linalg.inv(self.zi)),
                                                           np.transpose(self.yi))

    self.ff = self.simulation_function_xtx_btx.get_fn(self.xi, sea_cond=self.sea_condition, bb=self.bb)

    self.gi_old = self.gi
    self.hi_old = self.hi

    self.hi = jacobian(jacobian(self.simulation_function_xtx_btx.get_fn))(self.xi, sea_cond=self.sea_condition, bb=self.bb)
    #self.hi = self.simulation_function_xtx_btx.get_hessian_fn(self.xi, sea_cond=self.sea_condition, bb=self.bb)
    a = np.linalg.eigvals(self.hi)

    self.gi = np.subtract(np.matmul(self.hi, self.xi.transpose()),
                          jacobian(self.simulation_function_xtx_btx.get_fn)(self.xi, sea_cond=self.sea_condition, bb=self.bb))

    self.yi = self.yi + self.gi - self.gi_old
    self.zi = self.zi + self.hi - self.hi_old

    self.ratio = np.matmul(self.yi, np.linalg.inv(self.zi))
    self.ratio_evol.append(self.ratio)
    self.zi_evol.append(self.zi)


    print(f"Node ID: {self.node_id} -  Updating data ended!\n")

    return

def has_result_founded(self):
    """This method will check and verify if the calculated xi in this node has sufficiently converged. If for the
    last calculated xi, the difference between xi and x(i-1) is less than minimum accepted divergence that has
    been provided by the user, then it would be considered that calculated xi has enough convergence to its
    target value. """
    self.is_convergence_sufficient = False
    if len(self.all_calculated_xis) > 100:
        self.is_convergence_sufficient = True
        for i in range(100):
            for j in range(self.all_calculated_xis[-(i + 1)].size):
                if abs(abs(self.all_calculated_xis[-(i + 1)][j]) - abs(
                        self.all_calculated_xis[-(i + 1) - 1][j])) > self.minimum_accepted_divergence:
                    self.is_convergence_sufficient = False
    return self.is_convergence_sufficient




def Message(node_id, sigma_yi, sigma_zi):
    msg = {'node_id': node_id,
           'sigma_yi': sigma_yi,
           'sigma_zi': sigma_zi}
    return msg

# parameters

epsilon = 0.001
min_accepted_divergence = 0.02
c = 0.00000001
MAX_ITER = 10000
IPM_cycle = 1
cycle = 0
bb = 10
DONE = False

x0 = np.array([8, 800, 16, 170]) # initial point

cfn = CostFunction.CostFunction() # cost function


y0_prev = np.zeros(x0.size)
z0_prev = np.zeros((x0.size, x0.size))


# outer loop

while not DONE:
    pass