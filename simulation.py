import torch
import numpy as np





#### This file contains two classes, Forward simulation and Backward simulation. ###


### The simulation class is the base class for both forward and backward simulations.
### It contains some common functions that both simulations share.
class Simulation():
    ### This is a helper class, Not inteded to be used on its own.
     
    # Function to run a simulation based on given adjacency matrix and return histories.
    def run_simulation(self, adjacency_matrix):

        """
            Inputs:

                adjacency_matrix : Takes in an adjacency matrix.
                averaging_runs : No. of runs experiment is repeated, for smoothin/noise removal.


            Outputs:

                reward_average_history: Average reward each agent received from each arm.
                choice_average_history: Average number of times each agent picked each arm.

        
        """
        
        if ((self.num_agents, self.num_agents) != adjacency_matrix.shape):
            raise ValueError("Adjacenct matrix is of the wrong shape! Check number of agents!")


        reward_averaged_history = torch.zeros(self.num_agents, self.num_arms, device=self.device, requires_grad=True)
        choice_averaged_history = torch.zeros(self.num_agents, self.num_arms, device=self.device, requires_grad=True)

        for run in range(self.averaging_runs):

            history = torch.zeros(1, self.num_agents, self.num_arms)

            for epoch in range(1, self.max_epochs):

                performance_estimate = history.sum(dim=0) / (history.sum(dim=0).sum(dim=1).unsqueeze(1) + 1)
                arm_pull_count = history.bool().sum(dim=0).int()
                choice = torch.argmax(performance_estimate + self.epsilon * torch.sqrt(np.log(epoch)/(1+arm_pull_count)) +  (1/ self.num_agents) * adjacency_matrix @ performance_estimate, dim=1)
                
                rewards = torch.normal(mean=self.arm_performance[choice, 0], std=torch.abs(self.arm_performance[choice, 1]))


                current_epoch_record = torch.zeros(1, self.num_agents, self.num_arms, device=self.device)

                for i in range(self.num_agents):
                    for j in range(self.num_arms):
                        if j == choice[i]:
                            current_epoch_record[0, i, j] = rewards[i]


                history = torch.cat([history, current_epoch_record], dim=0)

            reward_history = history
            choice_history = history.bool().sum(dim=0).float()

            reward_averaged_history = reward_averaged_history + reward_history
            choice_averaged_history = choice_averaged_history + choice_history

        reward_averaged_history = reward_averaged_history / self.averaging_runs
        choice_averaged_history = choice_averaged_history / self.averaging_runs
            
        return reward_averaged_history, choice_averaged_history
    

    def save_state(self):

        """
        Saves the state of the class into a dict.

            Inputs: None,
            Ouputs: state_dict, containing all settings
        """

        state_dict = {
        "device" : self.device ,
        "num_agents" : self.num_agents ,
        "num_arms" : self.num_arms ,
        "arm_performance" : self.arm_performance ,
        "averaging_runs" : self.averaging_runs ,
        "max_epochs" : self.max_epochs ,
        "influence_weight" : self.influence_weight ,
        "epsilon" : self.epsilon ,
        }


        return state_dict

    def load_state(self, state_dict):

        """
            Loads settings from agiven state dict.
        
            Inputs: state_dict, a dict containing all settings
            Outputs: None
        """

        try:
            self.device = state_dict["device"]
            self.num_agents = state_dict["num_agents"]
            self.num_arms = state_dict["num_arms"]
            self.arm_performance = state_dict["arm_performance"]
            self.averaging_runs = state_dict["averaging_runs"]
            self.max_epochs = state_dict["max_epochs"]
            self.influence_weight = state_dict["influence_weight"]
            self.epsilon = state_dict["epsilon"]

            print("All settings succesfully loaded")

        except:

            print("Error loading settings.")



class ForwardSimulation(Simulation):

    """
    Class to handle a forward simulation.
    Initialize, Run simulations and collect results.
    

    Methods:
        init -> Initializes all the required settings\n
        run_simulation -> Runs simulation with given adjacency matrix and returns reward and choice history.\n
        save_state -> Saves all the settings into a new dict and returns it.\n
        load_state -> Gets all settings from a dict.\n
    """

    def __init__(self, num_arms=10, num_agents=10, device="cpu", averaging_runs=100, max_epochs=1000):

        # Initializing the required settings

        #Averaging runs is the number of the each simulation is run, to average results and remove noise.

        self.device = device
        self.num_agents = num_agents
        self.num_arms = num_arms
        self.arm_performance = torch.rand(num_arms, 2, device=self.device)
        self.averaging_runs = averaging_runs
        self.max_epochs = max_epochs
        self.influence_weight = 0.5
        self.epsilon = 2



class BackwardSimulation(Simulation):

    """
    Class to handle a backward simulation.
    Initialize, run simulations, backprop, collect adjacency matrix.
    

    Methods:
        init -> Initializes all the required settings
        run_simulation -> Runs simulation with given adjacency matrix and returns reward and choice history .
        train -> Uses backprop to update adjacency matrix, returns adjacency matrix.
        load_state -> Gets all settings from a dict.
    """

    def __init__(self, state_dict, adjacency_matrix=None):

        if state_dict is not None:
            self.load_state(state_dict=state_dict)

        if adjacency_matrix is None:
            self.adjacenyM = torch.rand(self.num_agents, self.num_agents, device=self.device, requires_grad=True)
        else:
            if adjacency_matrix.requires_grad == False:
                raise ValueError("The adjacency matrix must have requires_grad = True!")
            self.adjacenyM = adjacency_matrix

        


    def train(self, true_reward_history, true_choice_history, train_epochs=100):
        
        optimizer = torch.optim.AdamW([self.adjacenyM], lr=0.01)

        loss_history = []

        for epoch in range(train_epochs):

            rwrd_history, choice_history = self.run_simulation(self.adjacenyM)

            loss = torch.nn.functional.mse_loss(choice_history, true_choice_history)

            optimizer.zero_grad(set_to_none=True)

            loss.backward(retain_graph=True)

            optimizer.step()

            print(f"Epoch {epoch}, loss: {loss.item()}")
            loss_history.append(loss.item())


        return loss_history
    





        
    