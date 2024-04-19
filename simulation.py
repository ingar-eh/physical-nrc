import time

from Node import *
from async_simulation import *

import autograd.numpy as np
from autograd import jacobian
import numpy as np
import networkx as nx
import SimulationSpecification
import SimulationFunctionXTX_BTX


from NR_cntrsol import centralized_solution

import matplotlib.pyplot as plt

def Message(node_id, sigma_yi, sigma_zi, msg_rel):
    msg = {'node_id': node_id,
           'sigma_yi': sigma_yi,
           'sigma_zi': sigma_zi,
           'msg_rel': msg_rel}
    return msg


# ######## Experiment parameters #######
x0 = np.array([6, 750, 16]) #np.array([12.3, 1069.7, 24.01])  # initial point
DONE = True                  # flag for ending the experiment
IPM_cycle = 1               # number of interior-point-method cycles
cycle = 0                    # counter of the cycles done
bb = 5                   # steepness of the barrier functions


# needed class for simulation
sim_spec = SimulationSpecification.SimulationSpecification(x0)
random_selection = random_selection(sim_spec.number_of_nodes)
simulationFunction = SimulationFunctionXTX_BTX.SimulationFunctionXTX_BTX()

# ###### Useful constant and flag ########
sim_spec.initial_condition(x0)  # update initial point
agent_identifier = [i for i in range(sim_spec.number_of_nodes)]
epsilon = sim_spec.epsilon
convergence = np.zeros(sim_spec.number_of_nodes)
MAX_ITER = sim_spec.MAX_ITER
x0_i = np.zeros((sim_spec.number_of_nodes, IPM_cycle, x0.size))
y0_prev = np.zeros((sim_spec.number_of_nodes, x0.size))
z0_prev = np.zeros((sim_spec.number_of_nodes, x0.size, x0.size))

# ######## GENERATE NETWORK ##########
is_graph_needed = True
p = sim_spec.p  # probability of node connection
with open(f"network_topology.csv") as file_name:
    graph_matrix = np.loadtxt(file_name, delimiter=",")
network_graph = nx.from_numpy_matrix(graph_matrix)
'''while is_graph_needed:
    network_graph = nx.gnp_random_graph(sim_spec.number_of_nodes, p)
    graph_matrix = nx.to_numpy_array(network_graph)
    is_graph_needed = False
    for i in range(len(graph_matrix)):
        sum = 0
        for j in range(len(graph_matrix[i])):
            sum += graph_matrix[i][j]
        if sum == 0:
            is_graph_needed = True
            break'''
nx.draw(network_graph)
plt.show()
number_of_neighbors = [np.sum(graph_matrix[i]) for i in range(sim_spec.number_of_nodes)]

# ########### OUTER LOOP ###################
while DONE:

    # ######### INIT #############
    iter = 0
    nodes = []

    for i in range(sim_spec.number_of_nodes):
        if cycle == 0:
            ###### MODIFY INITIAL CONDITION #####
            x0_i[i, 0, 0] = x0[0] + np.random.randint(-1, 1)
            x0_i[i, 0, 1] = x0[1] + np.random.randint(20, 50)
            x0_i[i, 0, 2] = x0[2] + np.random.randint(3, 5)

        node = Node(i,
                    x0_i[i, cycle],
                    sim_spec.epsilon,
                    sim_spec.c,
                    sim_spec.min_accepted_divergence,
                    graph_matrix[i],
                    bb,
                    simulationFunction
                    )
        # for the following cycle, initialize the derivatives from the previous last value
        '''if cycle > 0:
            node.initial_point_IPM(y0_prev[i], z0_prev[i])'''
        nodes.append(node)
    diff = x0_i[0,0] - x0_i[1,0]
    message_container = [[] for i in range(sim_spec.number_of_nodes)]

    buffer = [[] for i in range(sim_spec.number_of_nodes)]

    # reset convergence flags for the next iterations
    CONVERGENCE_FLAG = False
    convergence = np.zeros(sim_spec.number_of_nodes)

    # ########### INNER LOOP ###################
    iter = 1

    while not CONVERGENCE_FLAG:
        # randomly activate some agents
        id_of_agent_activated = random_selection.persistent_communication()
        #id_of_agent_activated = random.sample(agent_identifier, k=2)

        # usefull breakpoint to debug
        if iter == 10000:
            print(f"iter: {iter}")
        if iter%1000 == 0:
            print(f"iter: {iter}")
        if iter%100 == 0:
            print(f"iter: {iter}")


        # trasmit data from the randomly activated agents
        for i in id_of_agent_activated:
            # Transmission
            nodes[i].transmit_data()
            # Broadcast
            message_container[i] = Message(nodes[i].node_id, nodes[i].sigma_yi, nodes[i].sigma_zi, nodes[i].msg_rel)
            for j in range(sim_spec.number_of_nodes):
                if nodes[i].adjacency_vector[j] == 1:
                    buffer[j].append(message_container[i])

        # receive message and update state
        for i in range(sim_spec.number_of_nodes):
            if len(buffer[i]) != 0:
                msg = buffer[i].pop()
                nodes[i].receive_data(msg)
                nodes[i].update_estimation(iter)
            if nodes[i].has_result_founded():
                convergence[i] = 1

        # check if convergence is reached
        if (convergence == 1).all() or iter >= MAX_ITER:
            CONVERGENCE_FLAG = True
            print(f"Reached convergence at iter:{iter}")
        else:
            iter += 1

    for j in range(sim_spec.number_of_nodes):
        print(f'            node_{j}:\n'
              f'sea condition:{nodes[j].sea_condition}\n'
              f'init:{nodes[j].all_calculated_xis[0]}\n'
              f'ending point:{nodes[j].xi}\n'
              f'----------------------------------------------------\n')

    #print(f'solution: {solution}')

    distances = [[] for a in range(sim_spec.number_of_nodes)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 8))
    for j in range(sim_spec.number_of_nodes):
        axs[0].plot(nodes[j].evolution_costfun, '-', label=f'J0_{j}')
        opt = nodes[j].xi
        '''for i in range(len(nodes[j].all_calculated_xis)):
            xi = np.array(nodes[j].all_calculated_xis[i])
            dst = np.sqrt(np.sum((opt - xi) ** 2))
            distances[j].append(dst)
        axs[2].plot(distances[j], '-', label=f'node_{j}')'''
        axs[1].plot(nodes[j].all_calculated_xis, label=f'node_{j}')
    axs[0].legend(loc='upper right', ncol=1)
    axs[0].set_title('Evolution of J0_i')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('cost function')
    axs[1].legend(loc='upper right', ncol=1)
    axs[1].set_title('Evolution xi')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Point-value evolution')
    axs[0].grid()
    axs[1].grid()
    plt.suptitle(f'cycle{cycle+1}')
    plt.savefig(f'multiple_plot_ss{cycle}.png')
    plt.show()

    # logarithmic plot of MSE and distance until optimum
    OptSol = centralized_solution(x0, bb)
    MSE = []
    distances = [[] for a in range(sim_spec.number_of_nodes)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 8))
    for k in range(iter-8):
        mse_k = 0
        for i in range(sim_spec.number_of_nodes):
            mse_k += ((np.abs(np.linalg.norm(nodes[i].all_calculated_xis[k])-np.linalg.norm(OptSol)))**2)/sim_spec.number_of_nodes
            dst = np.sqrt((OptSol-nodes[i].all_calculated_xis[k]) ** 2)
            distances[i].append(dst)
        MSE.append(mse_k)
    for i in range(sim_spec.number_of_nodes):
        axs[1].plot(distances[i], '-', label=f'node_{i}')
    axs[0].plot(MSE)
    #axs[0].legend(loc='upper right', ncol=1)
    axs[0].set_title('Mean Square Error')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('MSE')
    axs[1].legend(loc='upper right', ncol=1)
    axs[1].set_title('Distance until optimum found')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Distance until optimum')
    axs[0].grid()
    axs[1].grid()
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    plt.suptitle(f'cycle{cycle + 1}')
    plt.savefig(f'multiple_plot_log_ss{cycle}.png')
    plt.show()

    '''#fig, axs = plt.subplots(1, 1, figsize=(13, 8))
    for j in range(sim_spec.number_of_nodes):
        plt.plot(nodes[j].ratio_evol[0:100], label=f'g/h_{j}')
        #axs[1].plot(nodes[j].zi_evol[0:2000], label=f'z_{j}')
    plt.legend(loc='upper right', ncol=1)
    #axs[0].set_title('Evolution of ratio consensus')
    #axs[0].set_xlabel('Iteration')
    #axs[0].set_ylabel('Sum(g)/Sum(h)')
    #axs[1].legend(loc='upper right', ncol=1)
    #axs[1].set_title('Evolution zi')
    #axs[1].set_xlabel('Iteration')
    #axs[1].set_ylabel('z_i')
    #axs[0].grid()
    #axs[1].grid()
    plt.savefig('consensus_signal.png')
    plt.show()'''

    ##### update initial condition and steepness for next cycle
    avg_res = np.zeros(x0.shape)
    if cycle+1 != IPM_cycle:
        for i in range(sim_spec.number_of_nodes):
            avg_res += (nodes[i].xi / sim_spec.number_of_nodes)
            x0_i[i, cycle+1] = nodes[i].xi
            y0_prev[i] = nodes[i].yi
            z0_prev[i] = nodes[i].zi
    x0 = avg_res
    bb = bb*1.5
    cycle += 1

    # termination condition
    if cycle == IPM_cycle:
        print(f'final result:\n')
        f_total = 0
        for i in range(sim_spec.number_of_nodes):
            f_total += nodes[i].ff
            print(f'agent[{i}]:{nodes[i].xi}\n')
        DONE = False
        print(f'IPM completed \n final cost: {f_total}')
