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

    def send_signal_with_timeout(self, preamble_path, payload_path, timeout=10):
        """
        Sends a signal using sdm.send_signal_file() with a specified timeout.

        Args:
            preamble_path: Path to the preamble file.
            payload_path: Path to the payload file.
            timeout: Time to wait before terminating the send operation.
        """

        def send_signal_logic():
            try:
                #signal_data = sdm.stream_load_samples(payload_path);
                #sdm.send_tx(self.session, signal_data)
                #sdm.expect(self.session, sdm.REPLY_REPORT, sdm.REPLY_REPORT_TX_STOP)
                sdm.send_stop(self.session)
                sdm.expect(self.session, sdm.REPLY_STOP)
                sdm.send_signal_file(self.session, preamble_path, payload_path)
                
            except Exception as e:
                print(f"Error sending signal: {e}")

        # Check if timeout is a number
        if not isinstance(timeout, (int, float)):
            raise TypeError("Timeout must be a number")


        # Create a process for sending the signal
        send_process = multiprocessing.Process(target=send_signal_logic)

        # Start the process
        send_process.start()

        # Wait for the process to finish or timeout
        send_process.join(timeout=timeout)

        # Terminate the process if still running
        if send_process.is_alive():
            print("Send signal timed out.")
            sdm.send_stop(self.session)
            sdm.expect(self.session, sdm.REPLY_STOP)
            send_process.terminate()
            send_process.join()

    def janus_transmit(self, node_id, sigma_yi, sigma_zi):
        shortened_preamble_path = '/home/ingar/sdmsh/preamble_shortened.raw'
        payload_paths = ['/home/ingar/janus-c-3.0.5/payload.raw', '/home/ingar/janus-c-3.0.5/payload2.raw']
        new_payload_base_path = '/home/ingar/sdmsh/payload_w_preamble'

        # Ensure sigma_yi and sigma_zi are numpy arrays with the expected sizes
        sigma_yi = np.asarray(sigma_yi).flatten()[:4]
        sigma_zi = np.asarray(sigma_zi).flatten()[:16]

        # Convert the arrays to a space-separated string without brackets
        sigma_yi_str = np.array2string(sigma_yi, separator=' ', precision=2).replace('[', '').replace(']', '')
        sigma_zi_str = np.array2string(sigma_zi, separator=' ', precision=2).replace('[', '').replace(']', '')
        padding = "01010101"

        # Create the formatted string with brackets around each component
        payload = f"[{node_id}] [{sigma_yi_str}] [{sigma_zi_str}] {padding}"

        # Ensure the payload is on a single line by removing any unintended newline characters
        payload = payload.replace("\n", "")

        with open('/home/ingar/janus-c-3.0.5/src/c/shared_string.txt', 'w') as file:
                file.write(payload)
        time.sleep(5)
        for index, payload_path in enumerate(payload_paths):
            new_payload_path = f"{new_payload_base_path}_{index + 1}.raw"
            # Read content from the preamble and payload files
            with open(shortened_preamble_path, 'rb') as file:
                shortened_payload_content = file.read()
            with open(payload_path, 'rb') as file:
                payload_content = file.read()

            # Combine and write the content
            combined_content = shortened_payload_content + payload_content

            with open(new_payload_path, 'wb') as file:
                file.write(combined_content)

            # Send the signal using a timeout function

            self.send_signal_with_timeout(shortened_preamble_path, new_payload_path, timeout=10)



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
        time.sleep(1)
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
        #with open(file_path, 'rb') as f:
            # Read the entire file content into a numpy array as uint16
           # data = np.fromfile(f, dtype=np.uint16)
        data = np.fromfile(file_path, dtype=np.uint16)
            # Convert the numtcppy array to bytes
        data_bytes = data.tobytes()
            
            # Send data over TCP
        self.socket.sendall(data_bytes)
        

    def janus_receive(self, listen_time):
        def remove_quotes_from_file(filename):
            with open(filename, 'r') as file:
                data = file.read().strip().replace('"', '')
            return data

        def extract_integer_from_string(string):
            start_index = string.find('[')
            end_index = string.find(']')
            integer_string = string[start_index + 1:end_index]
            integer_value = int(integer_string)
            return integer_value

        def extract_floats_from_string(string):
            start_index = string.find('[', string.find(']'))
            end_index = string.find(']', start_index)
            floats_string = string[start_index + 1:end_index]
            float_strings = floats_string.split()
            floats_array = [float(float_str) for float_str in float_strings]
            return floats_array

        def concat_files(file1_content, file2_content):
            if file1_content.startswith('['):
                return file1_content + file2_content, "File 1 starts with an open bracket"
            elif file2_content.startswith('['):
                return file2_content + file1_content, "File 2 starts with an open bracket"
            else:
                return file1_content + file2_content, "None of the files starts with an open bracket"

        def janus_receive_logic(queue):
            sdm.send_stop(self.session)
            sdm.expect(self.session, sdm.REPLY_STOP)
            try:
                # Session setup and data reception logic
                sdm.add_sink_membuf(self.session)
                sdm.add_sink(self.session, "/home/ingar/sdmsh/rx.raw")
                sdm.send_rx(self.session, 1500000)
                sdm.expect(self.session, sdm.REPLY_STOP)

                self.session.receive.data = sdm.get_membuf(self.session)
                self.filename = "/home/ingar/sdmsh/rx.raw"

                self.send_file_over_tcp(self.filename)
                with open(self.filename, 'w') as file:
                    pass

                time.sleep(1)

                file1_path = '/home/ingar/janus-c-3.0.5/src/cargo_data1.txt'
                file2_path = '/home/ingar/janus-c-3.0.5/src/cargo_data2.txt'

                file1_content = remove_quotes_from_file(file1_path)
                print("This is file1: ", file1_content)
                file2_content = remove_quotes_from_file(file2_path)
                print("This is file2 :", file2_content)

                with open(file1_path, 'w') as file:
                    pass
                with open(file2_path, 'w') as file:
                    pass

                result, _ = concat_files(file1_content, file2_content)

                ID = extract_integer_from_string(result)
                sigma_yj = extract_floats_from_string(result)

                third_bracket_start_index = result.rfind(']', 0, result.rfind(']'))
                third_bracket_floats = extract_floats_from_string(result[third_bracket_start_index:])
                sigma_zj = np.array(third_bracket_floats).reshape(4, 4)

                queue.put((ID, sigma_yj, sigma_zj))

            except Exception as e:
                print("An error occurred:", e)
                queue.put((None, None, None))

        queue = multiprocessing.Queue()
        receive_process = multiprocessing.Process(target=janus_receive_logic, args=(queue,))

        receive_process.start()
        receive_process.join(timeout=listen_time)

        if receive_process.is_alive():
            print("Janus receive timed out.")
            sdm.send_stop(self.session)
            sdm.expect(self.session, sdm.REPLY_STOP)
            receive_process.terminate()
            receive_process.join()
            return None, None, None

        try:
            return queue.get_nowait()
        except queue.Empty:
            print("No result returned from Janus receive.")
            return None, None, None





    def receive_data(self, listen_time):
        # receive data from transmitting node
        ID, sigma_yj, sigma_zj = self.janus_receive(listen_time)
        print("checking fi ID is somethin")
        if ID is None:
            print("Failed to receive valid data.")
            print(f"Receiving data ended!\n")
            return None, None, None

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
        if len(self.all_calculated_xis) > 50:
            self.is_convergence_sufficient = True
            for i in range(50):
                for j in range(self.all_calculated_xis[-(i + 1)].size):
                    if abs(abs(self.all_calculated_xis[-(i + 1)][j]) - abs(
                            self.all_calculated_xis[-(i + 1) - 1][j])) > self.minimum_accepted_divergence:
                        self.is_convergence_sufficient = False
        return self.is_convergence_sufficient

