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

class Game(tk.Tk):
    def __init__(self, c_size, n_cells, head_starting_position):
        super().__init__()
        self.canvas_size = c_size
        self.head_starting_position = head_starting_position # for reseting the env

        self.env = Environment(n_cells, self.canvas_size)
        self.snake = Agent(head_starting_position)

        # class can host graphics

        self.game_over = False
        self._direction_step = 0

        self._init_canvas()

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
        self.env.step(directions[self._direction_step], self.snake)
        self._direction_step += 1
        if self._direction_step == len(directions):
            print('out of directions')
            self.game_over = True
        self.render()

    def run(self):
        #self.draw_map_points()
        self.env.reset(self.head_starting_position)

        self.render()

        def cb():
            if not self.game_over:
                self._update()
            self.after(1000, cb)
        self.after(1000, cb)
        self.mainloop()


numberOfCells = 6
startingPosition = (0, 0)
directions = [Direction.DOWN, Direction.DOWN, Direction.DOWN, Direction.RIGHT, Direction.RIGHT, Direction.RIGHT, Direction.UP, Direction.LEFT]
#directions = [Direction.UP, Direction.UP, Direction.RIGHT]


game = Game(800, numberOfCells, startingPosition)
game.run()
