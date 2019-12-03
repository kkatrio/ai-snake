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
        self.stack.append(head_starting_position)
    
    def _update_head_position(self, direction):
        if direction == Direction.UP:
            self.head_j -= 1
        if direction == Direction.LEFT:
            self.head_i -= 1
        if direction == Direction.DOWN:
            self.head_j += 1
        if direction == Direction.RIGHT:
            self.head_i += 1

    def move_head(self, direction):
        # calculates the new head position, appends it, returns the new
        self._update_head_position(direction)
        print("head_i, head_j: ", self.head_i, self.head_j)
        self.stack.append((self.head_i, self.head_j))
        return (self.head_i, self.head_j)

    def erase_tail(self):
        self.stack.popleft()
        

    @property
    def head(self):
        # returns the position(index) of the top of the stack
        # it must always be > 0
        return self.stack[len(self.stack)-1]

    @property
    def tail(self):
        #print('property returns tail', self.stack[0])
        return self.stack[0]

    @property
    def size(self):
        return len(self.stack)
