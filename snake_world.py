import numpy as np
import random
from snake import Snake, Directions

class CellType:
    HEAD = 0
    BODY = 1
    EMPTY = 2
    FOOD = 3

AllDirections = [Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.EAST]
Direction_map = {
        AllDirections[0]: 'NORTH',
        AllDirections[1]: 'WEST',
        AllDirections[2]: 'SOUTH',
        AllDirections[3]: 'EAST'
    }

class Actions():
    CONTINUE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    action_size = 3

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Map():
    def __init__(self, nCells, canvas_size):
        self.numberOfCells = nCells # in each axis
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
    def __init__(self, nCells, worldSize):
        self.numberOfCells = nCells
        self.world_size = worldSize # pixels
        #self.map = Map(self.numberOfCells, self.world_size) # needed only for visualization
        self.snake = None
        self.current_direction = None

        # the state really
        self._cellType = np.empty((self.numberOfCells, self.numberOfCells), dtype=CellType)
        # env must be reset before used - otherwise the cells are empty - shape is ok though

        self.done = False
        # must have action space = Direction

    def __getitem__(self, indices):
        i, j = indices
        return self._cellType[(i, j)]

    def __setitem__(self, indices, cell_type):
        self._cellType[indices] = cell_type

    #@property
    #def get_map(self):
    #    return self.map

    @property
    def state(self):
        return self._cellType

    @property
    def state_size(self):
        return self._cellType.shape
    #must have property action size

    def reset(self, head_position, head_direction, food_position): # things needed here : body

        # setup the snake at its starting position
        self.snake = Snake(head_position) # initializing a snake here?
        self.current_direction = head_direction
        self.done = False

        # put empty cells
        self._cellType[:][:] = CellType.EMPTY

        food_i, food_j = food_position
        self._cellType[food_i][food_j] = CellType.FOOD

        # put head
        self._cellType[head_position[0]][head_position[1]] = CellType.HEAD
        return self._cellType
        # todo: _cellType maybe should be called cellState

        # quick & dirty body setup
        self._cellType[4][4] = CellType.BODY
        self._cellType[4][5] = CellType.BODY


    def step(self, action):
        # make body the previous head
        previous_head = self.snake.head
        self[previous_head] = CellType.BODY # updates env cells

        self.current_direction = self.decide_way(action)
        #print('action: ', action, 'decided direction: ', Direction_map[self.current_direction])

        # grow by moving the head
        new_head_position = self.snake.move_head(self.current_direction)

        if self.has_hit_wall(new_head_position):
            #print('---HAS HIT WALL---')
            # maaybe we need to erase the tail here as well
            self.done = True

        elif self.has_hit_own_body(new_head_position):
            #print('---HAS HIT OWN BODY---')
            self.done = True

        # if we did not find food, erase the tail == grow back
        elif self[new_head_position] is not CellType.FOOD:
            #print('did not find food, moving on')
            previous_tail = self.snake.tail
            self.snake.erase_tail()
            self[previous_tail] = CellType.EMPTY # updates env cells
            self[new_head_position] = CellType.HEAD # updates env cells

        # but if we found food, regenerate
        else:
            #print('found food, regenerating')
            self.regenerate_food()
            self[new_head_position] = CellType.HEAD # updates env cells
            #print('CELLTYPES: ')
            #print(self._cellType)

        # make head the new position
        #print("after doing a step, snake size: ", self.snake.size)

        # env has been updated, so we can return its state
        reward = 1
        return (self._cellType, reward, self.done)

#AllDirections = [Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.EAST]

    def decide_way(self, action):
        dindex = AllDirections.index(self.current_direction)
        #print('current direction : ', Direction_map[AllDirections[dindex]])
        if action == Actions.TURN_LEFT:
            return AllDirections[(dindex + 1) % 4]
        elif action == Actions.TURN_RIGHT:
            return AllDirections[dindex - 1]
        elif action == Actions.CONTINUE_FORWARD:
            return self.current_direction

    def regenerate_food(self):
        ri = random.randrange(0, self.numberOfCells);
        rj = random.randrange(0, self.numberOfCells);
        self._cellType[ri][rj] = CellType.FOOD

    def has_hit_wall(self, head_position):
        i, j = head_position
        return True if i < 0 or i >= self.numberOfCells or j < 0 or j >= self.numberOfCells else False

    def has_hit_own_body(self, head_position):
        return self._cellType[head_position] == CellType.BODY


