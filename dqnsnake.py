from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from snake_world import Directions
import random

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.experience = deque(maxlen=2000)
        self.layers = None # convolutional layers
        self.numberOfLayers = 4
        self.batch_size = 64
        self.epsilon = 1 # explore probability
        self.gamma = 0.95 # discount factor
        #self.start_train = 100 # needed??
        self.input_shape = self.state_size + (self.numberOfLayers, )
        self.model = self.build_network()

    def build_network(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            # channels last
            input_shape = self.state_size + (self.numberOfLayers, ) # input_shape
        ))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1)
        ))
        model.add(layers.Activation('relu'))

        # Dense layers.
        model.add(layers.Flatten())
        model.add(layers.Dense(256))
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(self.action_size))

        model.summary()
        model.compile(tf.keras.optimizers.RMSprop(), 'MSE')
        return model

    def reset_convolutional_layers(self):
        # reset layers between episodes
        self.layers = None

    def get_convolutional_layers(self, state):
        layer = np.copy(state) # take a snapshot of this state
        if self.layers is None:
            self.layers = deque([layer] * self.numberOfLayers)
        else:
            self.layers.append(layer)
            self.layers.popleft()

        full_state = np.expand_dims(self.layers, 0) # make it 4d : (1,H,W,C)
        #print('full state: ', full_state)
        rolled = np.rollaxis(full_state, 1, 4)
        return rolled

    def store_transition(self, state, action, reward, next_state, done):

        memory_item = np.concatenate([
            state.flatten(),
            np.array(action).flatten(),
            np.array(reward).flatten(),
            next_state.flatten(),
            1 * np.array(done).flatten()
        ])
        self.experience.append(memory_item)
        #if len(self.experience) == 5:
        #    print('diff: \n', self.experience[4] - self.experience[0])

    def get_action(self, state):
        # explore
        if(np.random.rand() < self.epsilon):
            #print('exploring...')
            return random.randrange(self.action_size)

        #print('exploiting...')
        q_function = self.model.predict(state) # q-value function
        #print('q_function in get_action: ', q_function)
        return np.argmax(q_function[0])

    def train(self):
        # extract
        batch_size = min(len(self.experience), self.batch_size)
        #print('batch_size: ', batch_size)
        batch_experience = np.array(random.sample(self.experience, batch_size))

        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = batch_experience[:, 0:input_dim]
        actions = batch_experience[:, input_dim]
        rewards = batch_experience[:, input_dim + 1]
        states_next = batch_experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = batch_experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        rewards = rewards.repeat(self.action_size).reshape((batch_size, self.action_size))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.action_size).reshape((batch_size, self.action_size))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = self.model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.action_size).reshape((batch_size, self.action_size))

        delta = np.zeros((batch_size, self.action_size))
        delta[np.arange(batch_size), actions] = 1
        #jprint('delta: ', delta)
        #print('actions: ', actions)
        #print('y - Q function: ', y)

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + self.gamma * (1 - episode_ends) * Q_next)
        #print('targets - q functions: \n', targets)
        loss = float(self.model.train_on_batch(states, targets))
        return loss


