import numpy as np
import random
from snake import Snake, Directions

class CellType:
    HEAD = 2
    BODY = 3
    EMPTY = 0
    FOOD = 1
    WALL = 4

AllDirections = [Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.EAST]

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
        return self._cellType[indices]

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
        self.snake = Snake(head_position) # initializing a snake here? sure
        self.current_direction = head_direction
        self.done = False

        # put empty cells
        self._cellType[:] = CellType.EMPTY

        food_i, food_j = food_position
        self._cellType[food_i, food_j] = CellType.FOOD

        # quick & dirty body setup
        self._cellType[5, 4] = CellType.BODY
        self._cellType[6, 4] = CellType.BODY
        # don't forget to put the body onto the actual snake too
        self.snake.append_body((5, 4))
        tail = self.snake.tail
        print('appending body in reset - tail: ', tail)
        self.snake.append_body((6, 4))
        tail = self.snake.tail
        print('appending body in reset - tail: ', tail)

        # quickly put the walls
        self._cellType[[0, -1], :] = CellType.WALL
        self._cellType[:, [0, -1]] = CellType.WALL

        # put head
        self._cellType[head_position[0], head_position[1]] = CellType.HEAD
        return self._cellType
        # todo: _cellType maybe should be called cellState


    def step(self, action):

        # before we make a turn and move the head, we save the current cell position:
        # we will need it if we will not die
        previous_head = self.snake.head

        #print('current direction: ', self.current_direction)
        self.current_direction = self.turn(action)
        #print('new direction: ', self.current_direction)
        new_head_position = self.snake.move_head(self.current_direction) # just appends a new cell to the snake stack, i.e. grows its length

        if self.has_hit_wall(new_head_position) or self.has_hit_own_body(new_head_position):
            # do we need to kiil the new head? length is increased
            self.done = True
            reward = -1
            #print('died -- returning')
            return (self._cellType, reward, self.done)

        # if we did not find food:
        if self[new_head_position] is CellType.EMPTY:
            # erase tail - snake moves
            previous_tail = self.snake.tail
            self.snake.erase_tail()
            self[previous_tail] = CellType.EMPTY
            reward = 0

        elif self[new_head_position] is CellType.FOOD:
            #print('found FOOD, regenerating')
            self.regenerate_food()
            reward = 1 * self.snake.size

        else:
            print('Something is wrong when taking a step, what happened?')
            assert(False)

        # update the state for head and body
        self[previous_head] = CellType.BODY
        self[new_head_position] = CellType.HEAD # only after we have checked for empty space or food
        assert(self.done is False)
        return (self._cellType, reward, self.done)

    def turn(self, action):
        dindex = AllDirections.index(self.current_direction)
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
        return True if self._cellType[head_position] == CellType.WALL else False

    def has_hit_own_body(self, head_position):
        return self._cellType[head_position] == CellType.BODY


