import numpy as np
import Node


class SimulationSpecification:
    """Data-holder for simulation specification. This class handles all the required user interactions in order to
    collect the simulation specification. """
    def __init__(self, x0):
        self.number_of_nodes = 2 # : list[Node] = 6
        self.x0 = x0
        self.epsilon = 0.001
        self.min_accepted_divergence = 0.02
        self.c = 0.00000001
        self.p = 0.5  # prob of node selection
        self.MAX_ITER = 10000
        self.dimension = self.x0.size

    def initial_condition(self, x0):
        self.x0 = x0