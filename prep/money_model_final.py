import matplotlib.pyplot as plt
import numpy as np
# from money_model import *
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)

class MoneyAgent(Agent):
    def __init__(self, unique_id, model, type_agent):
        super().__init__(unique_id, model)
        self.wealth = 1
        self.type_agent = type_agent

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()
    
    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        wealth_neighbors = [cellmate.wealth for cellmate in cellmates]
        augmented_wealth_neighbors = [neighbor + 0.5 for neighbor in wealth_neighbors]
        sum_wealth_neighbors = sum(augmented_wealth_neighbors)

        if len(cellmates) > 1:
            if self.type_agent == "eq":
                # wealth_neighbors = [cellmate.wealth for cellmate in cellmates]
                # augmented_wealth_neighbors = [neighbor + 0.5 for neighbor in wealth_neighbors]
                # sum_wealth_neighbors = sum(augmented_wealth_neighbors)
                chance_chosen = [1/(neighbor / sum_wealth_neighbors) for neighbor in augmented_wealth_neighbors]
                other = self.random.choices(cellmates, weights = chance_chosen, k = 1)
                other = other[0]
                other.wealth += 1
                self.wealth -= 1
            elif self.type_agent == "uneq":
                # wealth_neighbors = [cellmate.wealth for cellmate in cellmates]
                # augmented_wealth_neighbors = [neighbor + 0.5 for neighbor in wealth_neighbors]
                # sum_wealth_neighbors = sum(augmented_wealth_neighbors)
                chance_chosen = [(neighbor / sum_wealth_neighbors) for neighbor in augmented_wealth_neighbors]
                other = self.random.choices(cellmates, weights = chance_chosen, k = 1)
                other = other[0]
                other.wealth += 1
                self.wealth -= 1
            elif self.type_agent == "stand":
                other = self.random.choice(cellmates)
                other.wealth += 1
                self.wealth -= 1

class MoneyModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height, type):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self, type)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
types = ["eq", "stand", "uneq"]
averages = []
for type_agent in types:
    fixed_params = {"width": 10,
                "height": 10,
                'type': type_agent,
                }
                
    variable_params = {"N": [100]}

    batch_run = BatchRunner(MoneyModel,
                            variable_params,
                            fixed_params,
                            iterations=10,
                            max_steps=100,
                            model_reporters={"Gini": compute_gini})
    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    average = run_data['Gini'].mean()
    averages.append(average)
    print(run_data)
    print(average)

plt.bar(types, averages)
plt.savefig("averages_money_model.png")

