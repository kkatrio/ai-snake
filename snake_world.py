import numpy as np
import random

class CellType:
    HEAD = 0
    BODY = 1
    EMPTY = 2
    FOOD = 3

class Direction():
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


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
    def __init__(self, nCells, wSize):
        self.numberOfCells = nCells
        self.world_size = wSize # pixels
        self.map = Map(self.numberOfCells, self.world_size)

        # the state really
        self._cellType = np.empty((self.numberOfCells, self.numberOfCells), dtype=CellType)

        # env must be reset before used, ok? - otherwise the cells are empty - shape is ok though

        self.done = False
        # must have action space = Direction

    def __getitem__(self, indices):
        i, j = indices
        return self._cellType[(i, j)]

    def __setitem__(self, indices, cell_type):
        self._cellType[indices] = cell_type

    @property
    def get_map(self):
        return self.map

    @property
    def state(self):
        return self._cellType

    @property
    def state_size(self):
        return self._cellType.shape
    #must have property action size

    def reset(self, head_position): # things needed here probably for food and body maybe
        # put empty cells
        self._cellType[:][:] = CellType.EMPTY

        # food on (2,3)
        self._cellType[2][3] = CellType.FOOD

        # put head
        self._cellType[head_position[0]][head_position[1]] = CellType.HEAD

        return self._cellType

    def step(self, direction, snake):
        # make body the previous head
        previous_head = snake.head
        self[previous_head] = CellType.BODY # updates env cells

        # grow by moving the head
        new_head_position = snake.move_head(direction)

        if self.has_hit_wall(new_head_position):
            print('has hit wall')
            self.done = True

        elif self.has_hit_own_body(new_head_position):
            print('has hit own body')
            self.done = True

        # if we did not find food, erase the tail == grow back
        elif self[new_head_position] is not CellType.FOOD:
            previous_tail = snake.tail
            snake.erase_tail()
            self[previous_tail] = CellType.EMPTY # updates env cells
            self[new_head_position] = CellType.HEAD # updates env cells

        # but if we found food, regenerate
        else:
            self.regenerate_food()
            self[new_head_position] = CellType.HEAD # updates env cells

        # make head the new position
        self[new_head_position] = CellType.HEAD # updates env cells
        print("snake size: ", snake.size)

        # env has been updated, so we can return its state
        reward = 1
        return (self._cellType, reward, self.done)

    def regenerate_food(self):
        ri = random.randrange(0, self.numberOfCells);
        rj = random.randrange(0, self.numberOfCells);
        self._cellType[ri, rj] = CellType.FOOD

    def has_hit_wall(self, head_position):
        i, j = head_position
        return True if i < 0 or i >= self.numberOfCells or j < 0 or j >= self.numberOfCells else False

    def has_hit_own_body(self, head_position):
        return self._cellType[head_position] == CellType.BODY


