import tkinter as tk
from snake_world import Environment, Point
from snake import Agent

numberOfCells = 6

class Game(tk.Tk):
    def __init__(self, c_size):
        super().__init__()
        self.canvas_size = c_size
        self.width = self.canvas_size
        self.height = self.canvas_size
        self._init_canvas()
        self.env = Environment(numberOfCells, self.canvas_size)
        self.snake = Agent(numberOfCells)

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
        tail_position = self.snake.tail
        tail = envMap[tail_position]
        self._canvas.create_rectangle(tail.x, tail.y, tail.x + envMap.edge, tail.y + envMap.edge, fill='white', outline='')

    def _update_contents(self):
        if self.snake.size > 2: # erases tail just before move appends a new head
            self.erase_tail()
        self.snake.move(3)
        print('stack len after move:', self.snake.size)
        self.draw_head()

    def show(self):
        self.draw_map_points()
        self.draw_head() # draw the snake initially
        
        def cb():
            self._update_contents()
            self.after(1000, cb)
        self.after(100, cb)


        self.mainloop()

game = Game(800)
game.show()
