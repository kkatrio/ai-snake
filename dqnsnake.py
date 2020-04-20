from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from snake_world import Directions
import random

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.experience = deque(maxlen=2000)
        self.frames = None # convolutional layers
        self.numberOfChannels = 4
        self.batch_size = 64
        self.epsilon = 1 # explore probability
        self.gamma = 0.95 # discount factor
        #self.start_train = 100 # needed??
        self.model = self.build_network()

    def build_network(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            # channels last
            input_shape = self.state_size + (self.numberOfChannels, )
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

    def get_channels(self, frame):
        if self.frames is None:
            self.frames = deque([frame] * self.numberOfChannels)
        else:
            self.frames.append(frame)
            self.frames.popleft()

        full_state = np.expand_dims(self.frames, 0) # make it 4d : (1,H,W,C)
        return np.rollaxis(full_state, 1, 4)

    def store_transition(self, state, action, reward, next_state, done):
        # todo: flatten before storing - save memory?

        # state and next_state here are (1,H,W,C)
        self.experience.append((state, action, reward, next_state, done))

    def get_action(self, state):
        #if(np.random.rand() < self.epsilon):
        #    return random.randrange(self.action_size)

        q_function = self.model.predict(state) # q-value function
        print('q_function in get_action: ', q_function)
        return np.argmax(q_function[0])

    def train(self):
        #if len(self.experience) < self.start_train:
        #    return

        # extract 
        batch_size = min(len(self.experience), self.batch_size)
        print('batch_size: ', batch_size)
        batch = random.sample(self.experience, batch_size)

        rewards, actions, dones = [], [], []

        states_size = (batch_size, )  + self.state_size + (self.numberOfChannels, )
        states = np.zeros(states_size) # (NHWC)
        next_states = np.zeros(states_size)

        for i in range(batch_size):
            #assert batch[i][0].shape == states[i].shape
            states[i] = batch[i][0] #implicit conversion to float from int

            actions.append(batch[i][1])
            rewards.append(batch[i][2])

            #assert batch[i][3].shape == next_states[i].shape
            next_states[i] = batch[i][3]

            dones.append(batch[i][4])

        #assert len(rewards) == batch_size
        #assert len(actions) == batch_size
        #assert len(dones) == batch_size

        #print('rewards: ', rewards)
        #print('actions: ', actions)
        #print('dones: ', dones)
        #print('states: ', states)

        # calculate Q-fuctions
        # model has changed so functions of previous states are going to be different too
        Q_function = self.model.predict(states) # states shape: (batch_size, action_size)
        Q_function_next_state = self.model.predict(next_states)

        # calculate target using Q of next state and udpate Q with it
        for i in range(len(Q_function)): # for each sample
            # calculate target from experience
            if dones[i]:
                target = rewards[i] # if done, no reason to add value for next move
            else:
                target = rewards[i] + self.gamma * np.amax(Q_function_next_state[i]) # we take the action for which Q is max
            #print("target: ", target)
            #print('actions[i]: ', actions[i])
            # actions contains the action to take (0, 1, or 2) for each sample
            Q_function[i][actions[i]] = target
            #print('Q_function: ', Q_function)

            # Q <- Q + a(target - Q)
            # from the prediction we take the action tha maximizes the q_function
            #self.model.fit(states, Q_function, batch_size=self.batch_size, epochs=1, verbose=0)
            # todo: test with train_on_batch

        #print('states: \n', states, '\n Q funtion: \n', Q_function)
        loss = self.model.train_on_batch(states, Q_function)
        return loss

