from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from dqn.snake_world import Directions
import random

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DQNAgent():
    def __init__(self, state_size, action_size, deterministic=False, batch_size=64, memory_limit=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.experience = deque(maxlen=memory_limit)
        self.layers = None # convolutional layers
        self.numberOfLayers = 4
        self.batch_size = batch_size
        self.epsilon = 1 # explore probability
        self.gamma = 0.95 # discount factor
        #self.start_train = 100 # needed??
        self.input_shape = self.state_size + (self.numberOfLayers, )

        if(deterministic):
            self.deterministic = True
            tf.random.set_seed(1)
            np.random.seed(0)
        else:
            self.deterministic = False

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

        #model.summary()
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


    def quick_save(self, state):
        #memory_item = state.flatten()
        self.experience.append(state)

    @property
    def print_memory(self):
        #print('memory: \n:', self.experience[len(self.experience) - 1] - self.experience[0])
        print('memory: \n:', self.experience)

    def save_transition(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))

    def get_exploration_action(self):
        return np.random.randint(self.action_size)

    def get_action(self, state):
        # explore
        if(np.random.random() < self.epsilon):
            #print('exploring...')
            return np.random.randint(self.action_size)

        #print('exploiting...')
        q_function = self.model.predict(state) # q-value function
        #print('q_function in get_action: ', q_function)
        return np.argmax(q_function[0])

    def train(self):
        # extract
        batch_size = min(len(self.experience), self.batch_size)
        #print('batch_size: ', batch_size)

        if (self.deterministic):
            experience_array = np.array(self.experience)
            batch_experience = experience_array[-batch_size:]
        else:
            batch_experience = np.array(random.sample(self.experience, batch_size))

        #for i in batch_experience:
        #    print('experience element: \n', i)

        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = np.zeros((batch_size, ) + self.input_shape, dtype=int)
        next_states = np.zeros((batch_size, ) + self.input_shape, dtype=int)
        for i in range(batch_size):
            states[i] = batch_experience[i, 0]
            next_states[i] = batch_experience[i, 3]

        actions = batch_experience[:, 1]
        rewards = batch_experience[:, 2]
        done_flags = batch_experience[:, 4]

        actions = np.cast['int'](actions)
        rewards = np.cast['int'](rewards)
        done_flags = np.cast['int'](done_flags)

        Q_function = self.model.predict(states)
        Q_function_next = self.model.predict(next_states)

        #print('Q: ', Q_function)
        #print('Qnext: ', Q_function_next)

        targets = rewards + self.gamma * (1 - done_flags) * np.amax(Q_function_next, axis=1)
        #assert(targets.shape == (batch_size, ))

        Q_function[np.arange(batch_size), actions] = targets
        #assert(Q_function.shape == (batch_size, 3))

        loss = float(self.model.train_on_batch(states, Q_function))
        return loss
