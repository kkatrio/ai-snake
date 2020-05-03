from collections import deque

class Directions():
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3

class Snake():
    def __init__(self, head_starting_position):
        self.head_i, self.head_j = head_starting_position
        self.stack = deque()
        # initialize head
        self.stack.append(head_starting_position)

    def _update_head_position(self, direction):
        # j is the second index in the array state -> move horizontaly
        # i is the first -> move vertically
        if direction == Directions.NORTH:
            self.head_i -= 1
        if direction == Directions.WEST:
            self.head_j -= 1
        if direction == Directions.SOUTH:
            self.head_i += 1
        if direction == Directions.EAST:
            self.head_j += 1

    def move_head(self, direction):
        # calculates the new head position, appends it, returns the new
        self._update_head_position(direction)
        #print("new head_i, head_j: ", self.head_i, self.head_j)
        self.stack.append((self.head_i, self.head_j))
        return (self.head_i, self.head_j)

    def append_body(self, body_position):
        self.stack.appendleft(body_position)

    def erase_tail(self):
        self.stack.popleft()

    def lies_on_position(self, position):
        return True if position in self.stack else False

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
