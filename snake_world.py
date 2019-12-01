import numpy as np

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Map():
    def __init__(self, nCells, canvas_size):
        self.numberOfCells = nCells
        self.numberOfPoints = self.numberOfCells + 1
        self.edge = canvas_size / self.numberOfCells

        self.map = np.empty((self.numberOfPoints, self.numberOfPoints), dtype = Point)
        self.create_coords()


    def create_coords(self):
        for i in range(self.numberOfPoints):
            for j in range(self.numberOfPoints):
                self.map[i, j] = Point(i * self.edge, j * self.edge)

    def __getitem__(self, indices):
        i, j = indices
        return self.map[i, j]




class Environment:
    def __init__(self, nCells, wSize):
        self.numberOfCells = nCells
        self.world_size = wSize # pixels
        self.map = Map(self.numberOfCells, self.world_size)

        # state: for every cell: [head, body, empty, food]
        self.state_size = (self.numberOfCells, self.numberOfCells, 4)

        self.cells = np.empty((self.numberOfCells, self.numberOfCells))

    @property
    def get_map(self):
        return self.map


'''
    def reset(self):

        state = np.zeros(state_size, dtype=int) # todo: bool

        # put walls
        #
        # put food
        #
        # put head and a body

        return state
'''
