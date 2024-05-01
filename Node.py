import autograd.numpy as np
from autograd import jacobian
import sdm
import time
import socket
import multiprocessing

class Node:

    def __init__(self, node_id: int, x0: np.array, epsilon: float, c: float,
                 minimum_accepted_divergence: float,
                 neighboring_nodes: np.array,
                 bb: float,
                 CostFunction, socket, session=None):


        self.node_id = node_id
        self.epsilon = epsilon
        self.xi = x0
        self.bb = bb
        self.socket = socket
        self.session = session

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

    def janus_transmit(self, node_id, sigma_yi, sigma_zi):
        shortened_preamble_path = '/home/ingar/sdmsh/preamble_shortened.raw'
        payload_paths = ['/home/ingar/janus-c-3.0.5/payload.raw', '/home/ingar/janus-c-3.0.5/payload2.raw']
        new_payload_base_path = '/home/ingar/sdmsh/payload_w_preamble'

        # Ensure sigma_yi and sigma_zi are numpy arrays with the expected sizes
        sigma_yi = np.asarray(sigma_yi).flatten()[:4]  # Ensures only the first 4 values are used
        sigma_zi = np.asarray(sigma_zi).flatten()[:16]  # Ensures only the first 16 values are used

        # Convert the arrays to a space-separated string without brackets
        sigma_yi_str = np.array2string(sigma_yi, separator=' ', max_line_width=np.inf, precision=2).replace('[', '').replace(']', '')
        sigma_zi_str = np.array2string(sigma_zi, separator=' ', max_line_width=np.inf, precision=2).replace('[', '').replace(']', '')

        # Form the payload
        payload = f"{node_id} {sigma_yi_str} {sigma_zi_str}"
        print("Sending: ", payload)

        for index, payload_path in enumerate(payload_paths):
            new_payload_path = f"{new_payload_base_path}_{index + 1}.raw"
            
            # Write the combined payload to the file
            with open('/home/ingar/janus-c-3.0.5/src/c/shared_string.txt', 'w') as file:
                file.write(payload)
            time.sleep(0.25)
            # Read and combine the original payloads
            with open(shortened_preamble_path, 'rb') as file:
                shortened_payload_content = file.read()
            with open(payload_path, 'rb') as file:
                payload_content = file.read()

            combined_content = shortened_payload_content + payload_content

            # Write the combined content to the new payload file
            with open(new_payload_path, 'wb') as file:
                file.write(combined_content)
            #print(f'New file created at: {new_payload_path}')

            # Additional operations like setting up sessions and configurations can be added here if needed.
            #sdm.send_ref_file(self.session, shortened_preamble_path)
            sdm.send_signal_file(self.session, shortened_preamble_path, new_payload_path)
            
        return


    def transmit_data(self):
        """This method update the yi and zi in each iteration and create a message including the new updated yi and
        zi. Finally the new message will be broadcast to all neighbors of this node. """
        #print(f"transmitting data started!\n")

        self.yi = (1 / (self.number_of_neighbors + 1)) * self.yi
        self.zi = (1 / (self.number_of_neighbors + 1)) * self.zi

        self.sigma_yi = self.sigma_yi + self.yi
        self.sigma_zi = self.sigma_zi + self.zi

        # broadcast values to neighboring nodes
        self.janus_transmit(self.node_id, self.sigma_yi, self.sigma_zi)
        # code goes here
        time.sleep(0.5)
       # print(f"transmitting data ended!\n")
        return

    def send_file_over_tcp(self, file_path):
        """
        Sends the content of a file over an established TCP connection as uint16 values.
        
        Args:
        s (socket.socket): An established socket connection.
        file_path (str): The path to the file to send.
        """
        # Open the file in binary read mode
        with open(file_path, 'rb') as f:
            # Read the entire file content into a numpy array as uint16
            data = np.fromfile(f, dtype=np.uint16)
            
            # Convert the numpy array to bytes
            data_bytes = data.tobytes()
            
            # Send data over TCP
            self.socket.sendall(data_bytes)
        

    def janus_receive(self, listen_time):
        def janus_receive_logic(queue):
            try:
                ref_sample_number = 1024 * 16
                usbl_sample_number = 51200
                ref_signal_file = "/home/ingar/sdmsh/preamble_shortened.raw"

                # Existing session setup and configuration code...

                sdm.add_sink_membuf(self.session)
                sdm.add_sink(self.session, "/home/ingar/sdmsh/rx.raw")
                sdm.send_rx(self.session, 2000896)
                sdm.expect(self.session, sdm.REPLY_STOP)

                self.session.receive.data = sdm.get_membuf(self.session)

                self.filename = "/home/ingar/sdmsh/rx.raw"
                self.send_file_over_tcp(self.filename)
                time.sleep(1)

                file_path = ['/home/ingar/janus-c-3.0.5/src/cargo_data1.txt', '/home/ingar/janus-c-3.0.5/src/cargo_data2.txt']

                try:
    # Reading contents of the files.
                    with open('/home/ingar/janus-c-3.0.5/src/cargo_data1.txt', 'r') as file1:
                        cargo_data1 = file1.read().strip().replace('"', '')

                    with open('/home/ingar/janus-c-3.0.5/src/cargo_data2.txt', 'r') as file2:
                        cargo_data2 = file2.read().strip().replace('"', '')

                    # Check which file begins with integer zero and concatenate in appropriate order
                    first_char1 = cargo_data1[0] if cargo_data1 else ''
                    first_char2 = cargo_data2[0] if cargo_data2 else ''

                    if first_char1 == '0':
                        cargo_data = cargo_data1 + cargo_data2
                    elif first_char2 == '0':
                        cargo_data = cargo_data2 + cargo_data1
                    else:
                        # If neither file starts with zero, default to concatenating in the original order
                        cargo_data = cargo_data1 + cargo_data2

                    # Split the concatenated string into individual values
                    values = cargo_data.split()

                    if len(values) != 21:  # 1 integer + 20 floats
                        raise ValueError("Invalid number of values in the file")

                    ID = int(values[0])
                    sigma_yj = list(map(float, values[1:5]))
                    sigma_zj = np.array(list(map(float, values[5:]))).reshape((4, 4))

                    # Send the results back via a queue.
                    queue.put((ID, sigma_yj, sigma_zj))

                except ValueError as ve:
                    print("Error:", ve)
                    queue.put((None, None, None))

            except Exception as e:
                print("An error occurred:", e)
                queue.put((None, None, None))

        # Create a queue to communicate between processes.
        queue = multiprocessing.Queue()

        # Create a separate process for the janus_receive_logic function.
        receive_process = multiprocessing.Process(target=janus_receive_logic, args=(queue,))

        # Start the process.
        receive_process.start()

        # Wait for the process to complete with a timeout of 10 seconds.
        receive_process.join(timeout=listen_time)

        # If the process is still alive after the timeout, terminate it.
        if receive_process.is_alive():
            print("Janus receive timed out.")
            sdm.send_stop(self.session)
            receive_process.terminate()
            receive_process.join()
            return None, None, None

        # Otherwise, fetch the result from the queue.
        try:
            return queue.get_nowait()
        except queue.Empty:
            print("No result returned from Janus receive.")
            return None, None, None





    def receive_data(self, listen_time):
        print(f"receiving data started!\n")
        # receive data from transmitting node
        ID, sigma_yj, sigma_zj = self.janus_receive(listen_time)

        if ID is None:
            print("Failed to receive valid data.")
            print(f"Receiving data ended!\n")
            return None, None, None

        print("Received data:", sigma_yj, sigma_zj)
        # Assuming ID is used to index into lists, ensure it's valid:
        ID = 0

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
        return ID, sigma_yj, sigma_zj


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


