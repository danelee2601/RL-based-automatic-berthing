import sys, os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_excel('./Conventional_turn_of_VLCC_rudder_angle_35deg.xlsx')
df2 = pd.read_excel('./norm_coordinates.xlsx')

plt.plot(df1.iloc[:,0], df1.iloc[:, 1])
plt.plot(df2.iloc[:,0], df2.iloc[:, 1])
plt.grid()
plt.show()
