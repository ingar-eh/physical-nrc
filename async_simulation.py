import random

class random_selection():

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agent_list = [i for i in range(self.n_agents)]

    def persistent_communication(self):
        id_of_agent_activated = random.sample(self.agent_list, k=1)#random.randint(1, len(self.agent_list)))
        for i in id_of_agent_activated:
            self.agent_list.remove(i)
        if len(self.agent_list) == 0:
            self.agent_list = [i for i in range(self.n_agents)]
        return id_of_agent_activated


