import matplotlib.pyplot as plt
import numpy as np
from money_model import *

model = MoneyModel(50, 10, 10)
for i in range(100):
    model.step()

gini = model.datacollector.get_model_vars_dataframe()
gini.plot()
plt.show()
