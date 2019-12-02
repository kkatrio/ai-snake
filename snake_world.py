import numpy as np

class CellType:
    HEAD = 0
    BODY = 1
    EMPTY = 2
    FOOD = 3


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
   
    @property 
    def size(self):
        return self.numberOfCells


class Environment:
    def __init__(self, nCells, wSize, head_position):
        self.numberOfCells = nCells
        self.world_size = wSize # pixels
        self.map = Map(self.numberOfCells, self.world_size)

        # state: for every cell: [head, body, empty, food]
        self.state_size = (self.numberOfCells, self.numberOfCells, 4)

        self._cellType = np.empty((self.numberOfCells, self.numberOfCells), dtype=CellType)

        self.state = np.zeros(self.state_size, dtype=int) # todo: bool
        self.headPosition = head_position
        self.reset()

    def __getitem__(self, indices):
        i, j = indices
        return self._cellType[(i, j)]

    def __setitem__(self, indices, cell_type):
        self._cellType[indices] = cell_type



    @property
    def get_map(self):
        return self.map


    def reset(self):

        # put empty cells
        self.state[:][:][2] = 1
        self._cellType[:][:] = CellType.EMPTY

        # food on (2,3)
        self.state[2][3][3] = 1
        self.state[2][3][2] = 0 # food is not empty
        self._cellType[2][3] = CellType.FOOD

        # put head
        self.state[self.headPosition[0]][self.headPosition[1]][0] = 1
        self.state[self.headPosition[0]][self.headPosition[1]][2] = 0
        self._cellType[self.headPosition[0]][self.headPosition[1]] = CellType.HEAD

    def print_state(self):
        print('state: ', self.state)

    def has_hit_wall(self, head_position):
        i, j = head_position
        return True if i < 0 or i >= self.numberOfCells or j < 0 or j >= self.numberOfCells else False

