import CostFunction
from Node import Node
#import Node
import autograd.numpy as np
import socket
import sdm

# node specific
node_id = 0
neighboring_nodes = np.array([1]) # ID list

# parameters
epsilon = 0.5
min_accepted_divergence = 0.2
c = 0.00000001
MAX_ITER = 10000
IPM_cycle = 1
bb = 10
x0 = np.array([1, 1250, 14, 28]) # initial point

ip = "127.0.0.1"
port = 9998

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ip, port))

session = sdm.create_session("NRC", "192.168.0.199")
session.timeout = 2000

shortened_preamble_path = '/home/ingar/sdmsh/preamble_shortened.raw'
#sdm.send_config(session, 50, 0, 2, 0)
#sdm.send_ref_file(session, shortened_preamble_path)
#sdm.expect(session, sdm.REPLY_REPORT, sdm.REPLY_REPORT_CONFIG)

# initialize the node as an object using the Node class
node = Node(node_id,
            x0,
            epsilon,
            c,
            min_accepted_divergence,
            neighboring_nodes,
            bb,
            CostFunction.CostFunction(),
            s,
            session)


# outer loop
for cycle in range(IPM_cycle):
    # reset convergence flag for the next iterations
    CONVERGENCE_FLAG = False
    iter = 1

    # inner loop
    while not CONVERGENCE_FLAG:
        # transmission
        sdm.send_stop(session)
        node.transmit_data()
        print("DONE TRANSMITTING")

        # receive message and update state
        ID, sigma_yj, sigma_zj = node.receive_data()

        # Only update estimation if data received is valid
        if ID is not None and sigma_yj is not None and sigma_zj is not None:
            print("Iteration number: ", iter)
            node.update_estimation(iter)
        else:
            print("Invalid data received, skipping update estimation.")

        # check if convergence is reached
        if node.has_converged() or iter >= MAX_ITER:
            CONVERGENCE_FLAG = True
            print(f"Reached convergence at iter:{iter}")
            session.close()
        else:
            iter += 1

    # update initial condition and steepness for the next cycle
    node.x0 = node.xi
    node.bb *= 1.5
