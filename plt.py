import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('training_data')

epochs = data[:, 0]
steps = data[:, 1]
loss = data[:, 2]

f1 = plt.figure('loss')
ax1 = plt.gca()
sc1, = ax1.plot(epochs, loss)
sc1.set_marker('.')
sc1.set_markerfacecolor('b')
sc1.set_markeredgecolor('b')
sc1.set_color('b')

f2 = plt.figure('steps')
ax2 = plt.gca()
sc2, = ax2.plot(epochs, steps)
sc1.set_marker('.')
sc1.set_markerfacecolor('r')
sc1.set_markeredgecolor('r')
sc1.set_color('r')


plt.show()
