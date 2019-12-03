import time
import tkinter as tk
from snake_world import Environment, Point, CellType
from snake import Agent, Direction

class Colors:
    CELLTYPE = {
        CellType.HEAD : 'black',
        CellType.BODY : 'blue',
        CellType.EMPTY : 'white',
        CellType.FOOD : 'yellow'
    }


numberOfCells = 6
startingPosition = (0, 0)
directions = [Direction.DOWN, Direction.DOWN, Direction.DOWN, Direction.RIGHT, Direction.RIGHT, Direction.RIGHT, Direction.UP, Direction.LEFT]
#directions = [Direction.UP, Direction.UP, Direction.RIGHT]


class Game(tk.Tk):
    def __init__(self, c_size):
        super().__init__()
        self.canvas_size = c_size
        self.width = self.canvas_size
        self.height = self.canvas_size
        self._init_canvas()
        # todo: get information about initial state in a better way for both Env and Agent
        self.env = Environment(numberOfCells, self.canvas_size, startingPosition)
        self.snake = Agent(numberOfCells, startingPosition)
        self.game_over = False
        self._step = 0

    def _init_canvas(self):
        self._canvas = tk.Canvas(self,
                                 bg='white',
                                 width=self.canvas_size,
                                 height=self.canvas_size,
                                 highlightthickness=0)
        self._canvas.pack()

    def draw_map_points(self):
        envMap = self.env.get_map
        for i in range(envMap.numberOfPoints):
            for j in range(envMap.numberOfPoints):
                p = Point(envMap[(i, j)].x, envMap[(i, j)].y)
                self._canvas.create_rectangle(p.x - 1, p.y - 1, p.x + 1, p.y + 1, fill='blue', outline='')

    def _update_contents(self, direction):
        # make body the previous head
        previous_head = self.snake.head
        self.env[previous_head] = CellType.BODY

        # move head by growing
        new_head_position = self.snake.move_head(direction)

        if self.env.has_hit_wall(new_head_position):
            print('has hit wall')
            self.game_over = True
            return

        if self.env.has_hit_own_body(new_head_position):
            print('has hit own body')
            self.game_over = True
            return

        # if we did not find food, erase the tail == undo the grow
        if self.env[new_head_position] is not CellType.FOOD:
            previous_tail = self.snake.tail
            self.snake.erase_tail() # we cound make it grow on a peek maybe
            self.env[previous_tail] = CellType.EMPTY

        # but if we found food, regenerate
        else:
            self.env.regenerate_food()


        # make head the new position
        self.env[new_head_position] = CellType.HEAD
        print("snake size: ", self.snake.size)

    def render(self):
        envMap = self.env.get_map #coords

        # brutal but ok
        for i in range(envMap.size):
            for j in range(envMap.size):
                cell = envMap[(i, j)]
                cellType = self.env[(i, j)]
                color = Colors.CELLTYPE[cellType]
                self._canvas.create_rectangle(cell.x, cell.y, cell.x + envMap.edge, cell.y + envMap.edge, fill=color, outline='')

    def _update(self):
        self._update_contents(directions[self._step])
        self._step += 1
        if self._step == len(directions):
            print('out of directions')
            self.game_over = True
        self.render()

    def show(self):
        #self.draw_map_points()
        self.render()

        def cb():
            if not self.game_over:
                self._update()
            self.after(1000, cb)
        self.after(1000, cb)
        self.mainloop()

game = Game(800)
game.show()
