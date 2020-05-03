import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()
with open(args.filename) as file:
    data = np.loadtxt(args.filename)

epochs = data[:, 0]
steps = data[:, 1]
loss = data[:, 2]
rewards = data[:, 3]
fruits = data[:, 4]

f1 = plt.figure('loss -- '+args.filename)
ax1 = plt.gca()
sc1, = ax1.plot(epochs, loss)
sc1.set_marker('.')
sc1.set_markerfacecolor('b')
sc1.set_markeredgecolor('b')
sc1.set_color('b')

f2 = plt.figure('steps -- '+args.filename)
ax2 = plt.gca()
sc2, = ax2.plot(epochs, steps)
sc2.set_marker('.')
sc2.set_markerfacecolor('r')
sc2.set_markeredgecolor('r')
sc2.set_color('r')

f3 = plt.figure('rewards -- '+args.filename)
ax3 = plt.gca()
sc3, = ax3.plot(epochs, rewards)
sc3.set_marker('.')
sc3.set_markerfacecolor('g')
sc3.set_markeredgecolor('g')
sc3.set_color('g')

f4 = plt.figure('fruits -- '+args.filename)
ax4 = plt.gca()
sc4, = ax4.plot(epochs, fruits)
sc4.set_marker('.')
sc4.set_markerfacecolor('c')
sc4.set_markeredgecolor('c')
sc4.set_color('c')

plt.show()
