import matplotlib.pyplot as plt
import numpy as np

from money_model import *

model = MoneyModel(50, 10, 10)
for i in range(20):
    model.step()

agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content, x, y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.savefig("test2.png")
# (array([3., 0., 0., 5., 0., 0., 1., 0., 0., 1.]),
#  array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3. ]),
#  <a list of 10 Patch objects>)

# agent_wealth = [a.wealth for a in model.schedule.agents]
# plt.hist(agent_wealth)
# plt.savefig("test.png")