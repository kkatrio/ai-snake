#!/usr/bin/env python3

from tensorflow import keras
from dqn.agent import DQNAgent
from dqn.snake_world import Environment, CellType
import numpy as np
from collections import deque
import argparse
import tkinter as tk


class Colors:
    CELLTYPE = {
        CellType.HEAD : 'cyan',
        CellType.BODY : 'blue',
        CellType.EMPTY : 'white',
        CellType.FOOD : 'red',
        CellType.WALL : 'black'
    }

class Outline:
    CELLTYPE = {
        CellType.HEAD : 'cyan',
        CellType.BODY : 'grey',
        CellType.EMPTY : 'white',
        CellType.FOOD : 'orange',
        CellType.WALL : 'black'
    }

class TrainedAgent():

    def __init__(self, model_dir_name):
        self.trained_model = model_dir_name
        self.model = self._load_model()
        self.numberOfLayers = self.model.input_shape[3]
        self.layers = None

    # todo: free function?
    def _get_convolutional_layers(self, state):
        layer = np.copy(state)
        if self.layers is None:
            self.layers = deque([layer] * self.numberOfLayers)
        else:
            self.layers.append(layer)
            self.layers.popleft()

        full_state = np.expand_dims(self.layers, 0)
        rolled = np.rollaxis(full_state, 1, 4)
        return rolled

    def _load_model(self):
        return keras.models.load_model(self.trained_model)

    def choose_action(self, state):
        state = self._get_convolutional_layers(state)
        Q_function = self.model.predict(state)
        return np.argmax(Q_function[0])

class Runner(tk.Tk):

    def __init__(self, env, agent):
        super().__init__()

        self.env = env # to get the cell type from
        self.env_map = env.get_map # coords (array of Points)
        self.agent = agent
        self.canvas_size = env.world_size
        self._init_canvas()
        self.geometry("400x400+1800+300") # screen pixels

    def _init_canvas(self):
        self._canvas = tk.Canvas(self,
                                 bg='white',
                                 width=self.canvas_size,
                                 height=self.canvas_size,
                                 highlightthickness=0)
        self._canvas.pack()

    def render(self):
        for i in range(self.env_map.size):
            for j in range(self.env_map.size):
                cell = self.env_map[(i, j)]
                cellType = self.env[(i, j)]
                color = Colors.CELLTYPE[cellType]
                outline = Outline.CELLTYPE[cellType]
                self._canvas.create_rectangle(cell.x, cell.y, cell.x + self.env_map.edge, cell.y + self.env_map.edge, fill=color, outline=outline)

    def get_next_move(self):
        state = self.env.state
        # predict
        action = self.agent.choose_action(state)
        _, _, self.game_over = self.env.step(action)
        score = self.env.snake.size
        title = 'score: ' + str(score - 3) # initial length
        self.title(title)
        self.render()

        if not self.game_over:
            self.after(200, self.get_next_move)

    def run(self):
        self.game_over = False
        self.render()
        self.after(4000, self.get_next_move) # delay after initial state
        self.mainloop()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    args = parser.parse_args()

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    #foodPosition = (3, 6)
    agent = TrainedAgent(args.modelname) # todo: pass model
    env = Environment(numberOfCells, worldSize=400)
    state = env.reset(startingPosition)
    runner = Runner(env, agent)
    runner.run()


if __name__ == "__main__":
    main()
