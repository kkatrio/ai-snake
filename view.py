from tensorflow import keras
from dqnsnake.agent.dqnsnake import DQNAgent
from dqnsnake.agent.snake_world import Environment, CellType
import numpy as np
from collections import deque
import tkinter as tk


class Colors:
    CELLTYPE = {
        CellType.HEAD : 'cyan',
        CellType.BODY : 'blue',
        CellType.EMPTY : 'white',
        CellType.FOOD : 'red',
        CellType.WALL : 'brown'
    }

class TrainedAgent():

    def __init__(self):
        self.model = self._load_model()
        self.numberOfLayers = 4 # todo: avoid hardcoded
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
        return keras.models.load_model('trained_snake.model')

    def choose_action(self, state):
        state = self._get_convolutional_layers(state)
        Q_function = self.model.predict(state)
        return np.argmax(Q_function[0])


class Viewer(tk.Tk):

    def __init__(self, env, agent, pixels):
        super().__init__()

        self.env = env # to get the cell type from
        self.envMap = env.get_map # coords (array of Points)
        self.agent = agent
        self.canvas_size = pixels
        self._init_canvas()

    def _init_canvas(self):
        self._canvas = tk.Canvas(self,
                                 bg='white',
                                 width=self.canvas_size,
                                 height=self.canvas_size,
                                 highlightthickness=0)
        self._canvas.pack()

    def render(self):
        for i in range(self.envMap.size):
            for j in range(self.envMap.size):
                cell = self.envMap[(i, j)]
                cellType = self.env[(i, j)]
                color = Colors.CELLTYPE[cellType]
                self._canvas.create_rectangle(cell.x, cell.y, cell.x + self.envMap.edge, cell.y + self.envMap.edge, fill=color, outline='')


    def cb(self):
        
        if not self.game_over:
            state = self.env.state
            action = self.agent.choose_action(state)
            _, _, self.game_over = self.env.step(action)
            self.render()
        self.after(1000, self.cb)
   
    def run(self):

        self.game_over = False
        step = 0
        self.render()

        self.after(10, self.cb)
        self.mainloop()


def main():
    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    foodPosition = (3, 6)
    agent = TrainedAgent() # todo: pass model
    env = Environment(numberOfCells, worldSize=800)
    state = env.reset(startingPosition, foodPosition)
    viewer = Viewer(env, agent, 800)
    viewer.run()


if __name__ == "__main__":
    main()
