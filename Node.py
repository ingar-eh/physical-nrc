import autograd.numpy as np
from autograd import jacobian

class Node:

    def __init__(self, node_id: int, x0: np.array, epsilon: float, c: float,
                 minimum_accepted_divergence: float,
                 neighboring_nodes: np.array,
                 bb: float,
                 CostFunction):


        self.node_id = node_id
        self.epsilon = epsilon
        self.xi = x0
        self.bb = bb

        # list for storing evolution of signals
        self.all_calculated_xis = []
        self.evolution_costfun = []
        self.zi_evol = []


        self.cfn = CostFunction
        self.ff = CostFunction.get_fn(self.xi, bb=self.bb)

        self.neighboring_nodes = neighboring_nodes
        self.number_of_neighbors = neighboring_nodes.size
        self.minimum_accepted_divergence = minimum_accepted_divergence

        self.hi_old = np.eye(x0.size)
        self.hi = np.eye(x0.size)

        self.gi = np.zeros(x0.size).transpose()
        self.gi_old = np.zeros(x0.size).transpose()
        self.zi = np.eye(x0.size)
        self.yi = np.zeros(x0.size).transpose()

        self.cI = c * np.eye(x0.size) # variable to be check in order to ensure robustness and large basin of attraction

        self.sigma_yi = np.zeros(x0.size).transpose()  # counter of the total mass-y sent
        self.sigma_zi = np.zeros((x0.size, x0.size))   # counter of the total mass-z sent, matrix x0.size X x0.size

        self.rho_yj = np.zeros((self.number_of_neighbors, x0.size))  # counter of the total mass-y received from j, matrix_dim X xo.size
        self.rho_yj_old = np.zeros((self.number_of_neighbors, x0.size))
        self.rho_zj = np.zeros((self.number_of_neighbors, x0.size, x0.size))  # counter of the total mass-z received from j
        self.rho_zj_old = np.zeros((self.number_of_neighbors, x0.size, x0.size))


    def transmit_data(self):
        """This method update the yi and zi in each iteration and create a message including the new updated yi and
        zi. Finally the new message will be broadcast to all neighbors of this node. """
        print(f"transmitting data started!\n")

        self.yi = (1 / (self.number_of_neighbors + 1)) * self.yi
        self.zi = (1 / (self.number_of_neighbors + 1)) * self.zi

        self.sigma_yi = self.sigma_yi + self.yi
        self.sigma_zi = self.sigma_zi + self.zi

        # broadcast values to neighboring nodes
        
        # code goes here

        print(f"transmitting data ended!\n")
        return

    def receive_data(self):
        """This method will handle the reception of data from neighbors. Using the yi and zi from the messages,
        the yi and zi would be updated by this method."""
        print(f"receiving data started!\n")

        # receive data from transmitting node

        # code goes here

        # temporary data values
        ID = self.neighboring_nodes(0)
        sigma_yj = 0
        sigma_zj = 0

        # update old virtual mass received
        self.rho_zj_old[ID] = self.rho_zj[ID]
        self.rho_yj_old[ID] = self.rho_yj[ID]

        # update virtual mass of neighbor
        self.rho_yj[ID] = sigma_yj
        self.rho_zj[ID] = sigma_zj

        # update estimate
        self.yi = self.yi + self.rho_yj[ID] - self.rho_yj_old[ID]
        self.zi = self.zi + self.rho_zj[ID] - self.rho_zj_old[ID]

        print(f"Receiving data ended!\n")
        return

    def update_estimation(self, iter):
        """This method will calculate the next xi and also update the hi and gi by using the new xi. """
        print(f"Updating data started!\n")
        self.all_calculated_xis.append(self.xi)
        self.evolution_costfun.append(self.ff)
        while iter > len(self.evolution_costfun):
            self.all_calculated_xis.append(self.xi)
            self.evolution_costfun.append(self.ff)

        # check condition on z
        if (np.abs(np.linalg.eigvals(self.zi)) < np.linalg.eigvals(self.cI)).all():
            self.zi = self.cI

        self.xi = (1 - self.epsilon) * self.xi + np.matmul((self.epsilon * np.linalg.inv(self.zi)),
                                                               np.transpose(self.yi))

        self.ff = self.cfn.get_fn(self.xi, bb=self.bb)

        self.gi_old = self.gi
        self.hi_old = self.hi

        self.hi = jacobian(jacobian(self.cfn.get_fn))(self.xi, bb=self.bb)

        self.gi = np.subtract(np.matmul(self.hi, self.xi.transpose()),
                              jacobian(self.cfn.get_fn)(self.xi, bb=self.bb))

        self.yi = self.yi + self.gi - self.gi_old
        self.zi = self.zi + self.hi - self.hi_old

        self.zi_evol.append(self.zi)


        print(f"Updating data ended!\n")

        return

    def has_converged(self):
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


