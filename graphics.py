import tkinter as tk
from snake_world import Environment, Point, CellType
from snake import Agent

class Colors:
    CELLTYPE = {
        CellType.HEAD : 'black',
        CellType.BODY : 'blue',
        CellType.EMPTY : 'white',
        CellType.FOOD : 'yellow'
    }


numberOfCells = 6
startingPosition = (2, 1)

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

    def draw_head(self):
        envMap = self.env.get_map # todo: maybe class can hold the map
        head_position = self.snake.head
        head = envMap[head_position]
        self._canvas.create_rectangle(head.x, head.y, head.x + envMap.edge, head.y + envMap.edge, fill='black', outline='')
       
    def erase_tail(self):
        envMap = self.env.get_map # todo: maybe class can hold the map
        tail_position = self.snake.tail_previous
        tail = envMap[tail_position]
        self._canvas.create_rectangle(tail.x, tail.y, tail.x + envMap.edge, tail.y + envMap.edge, fill='white', outline='')

    def _update_env_state(self):
        # write on env's state
        head_position = self.snake.head
        self.env.state[head_position[0]][head_position[1]][0] = 1
        self.env[head_position] = CellType.HEAD

        try: # todo: replace with has_eaten flag
            tail_position = self.snake.tail_previous
            self.env.state[tail_position[0]][tail_position[1]][2] = 1 # empty
            self.env[tail_position] = CellType.EMPTY
        except:
            pass

    def _update_contents(self):
        self.snake.move(3) # move also saves the previous tail
        print('stack len after move:', self.snake.size)


    def render(self):
        envMap = self.env.get_map #coords

        # brutal but ok
        for i in range(envMap.size):
            for j in range(envMap.size):
                cell = envMap[(i, j)]
                cellType = self.env[(i, j)]
                print('cellType: ', cellType)
                color = Colors.CELLTYPE[cellType]
                self._canvas.create_rectangle(cell.x, cell.y, cell.x + envMap.edge, cell.y + envMap.edge, fill=color, outline='')


    def show(self):
        self.draw_map_points()
        self.render()
        
        def cb():
            self._update_contents()
            self._update_env_state()
            self.render()
            self.after(1000, cb)
        self.after(100, cb)


        self.mainloop()

game = Game(800)
game.show()
