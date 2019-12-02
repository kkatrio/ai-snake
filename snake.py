from collections import deque

class Direction():
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


class Agent():
    def __init__(self, numberOfCells, head_starting_position):
        self.numberOfCells = numberOfCells
        self.head_i, self.head_j = head_starting_position # tuple
        self.stack = deque()
        # initialize head
        self.stack.append((self.head_i, self.head_j))
        self.last_tail = None
    
    def _update_head_position(self, direction):
        if direction == Direction.UP:
            if self.head_j < self.numberOfCells - 1: # to be replaced by hitting the wall
                self.head_j = 0
            else:
                self.head_j += 1
        if direction == Direction.LEFT:
            if self.head_i <= self.numberOfCells - 1:
                self.head_i = 0
            else:
                self.head_i += 1
        if direction == Direction.DOWN:
            if self.head_j >= self.numberOfCells - 1:
                self.head_j = 0
            else:
                self.head_j += 1
        if direction == Direction.RIGHT:
            if self.head_i >= self.numberOfCells - 1:
                self.head_i = 0
            else:
                self.head_i += 1


    def move(self, direction):
        #move head
        self._update_head_position(direction)
        print("head_i, head_j: ", self.head_i, self.head_j)
        self.stack.append((self.head_i, self.head_j))

        if len(self.stack) > 3: # todo: when not eaten food
            # when we erase from the stack, we must repaint first too!
            self.last_tail = self.stack[0] # save the tail just before we move for it
            self.stack.popleft()
        

    @property
    def head(self):
        # returns the position(index) of the top of the stack
        return self.stack[len(self.stack)-1]
    
    @property
    def tail_previous(self):
        # returns previous position of the tail
        #print('property returns tail', self.stack[0])
        return self.last_tail

    @property
    def size(self):
        return len(self.stack)
