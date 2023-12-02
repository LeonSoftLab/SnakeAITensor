import pygame
from random import randint, choice

# Global settings
SIZE_IMG = 25
WIDTH, HEIGHT = 20, 20  # Map
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
GREEN = (40, 225, 70)
RED = (255, 0, 0)
FPS = 60
VECTORS = ("left", "up", "right", "down")
direction_mapping = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1)
}


def gen_x_y():
    return randint(0, WIDTH - 1), randint(0, HEIGHT - 1)


def get_vector_from_turn(turn, current_vector):
    if turn == "left":
        if current_vector == "left":
            return "down"
        elif current_vector == "up":
            return "left"
        elif current_vector == "right":
            return "up"
        elif current_vector == "down":
            return "right"
    elif turn == "right":
        if current_vector == "left":
            return "up"
        elif current_vector == "up":
            return "right"
        elif current_vector == "right":
            return "down"
        elif current_vector == "down":
            return "left"
    else:
        return current_vector


# Food Object
class Food:
    def __init__(self, screen, x, y, color, size):
        self._screen = screen
        self._color = color
        self._size = size
        self.x = x
        self.y = y

    def regen(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        pygame.draw.rect(self._screen, self._color,
                         pygame.Rect(self.x * self._size, self.y * self._size, self._size, self._size)
                         )

    def clear(self, back_color):
        pygame.draw.rect(self._screen, back_color,
                         pygame.Rect(self.x * self._size, self.y * self._size, self._size, self._size)
                         )

    def check(self, x, y):
        # Check food position
        return self.x == x and self.y == y


# Snake Object
class Snake:
    def __init__(self, screen, x, y, color, size_block, vector):
        self._screen = screen
        self._vector = vector
        self._size_block = size_block
        self._color = color
        # add head
        self.body = [{"x": x, "y": y}]

    @property
    def head(self):
        return self.body[0]

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vector):
        self._vector = vector

    def _move_block(self, block, x, y):
        prev_x, prev_y = block["x"], block["y"]
        block["x"] = x
        block["y"] = y
        return prev_x, prev_y

    def move(self, vector=None):
        if vector:
            self._vector = vector
        # Move a head
        delta_x, delta_y = direction_mapping[self._vector]
        prev_x, prev_y = self.body[0]["x"], self.body[0]["y"]
        self.body[0]["x"] += delta_x
        self.body[0]["y"] += delta_y
        # Move a body
        for i in range(1, len(self.body)):
            prev_x, prev_y = self._move_block(self.body[i], prev_x, prev_y)
        return prev_x, prev_y

    def grow(self, x, y):
        last_block = self.body[-1].copy()
        last_block["x"] = x
        last_block["y"] = y
        self.body.append(last_block)

    def draw(self):
        for block in self.body:
            pygame.draw.rect(self._screen, self._color, pygame.Rect(block["x"] * self._size_block,
                                                                    block["y"] * self._size_block,
                                                                    self._size_block,
                                                                    self._size_block))

    def clear(self, back_color):
        for block in self.body:
            pygame.draw.rect(self._screen, back_color, pygame.Rect(block["x"] * self._size_block,
                                                                   block["y"] * self._size_block,
                                                                   self._size_block,
                                                                   self._size_block))

    def check_pos(self, x, y):
        for block in self.body:
            if block["x"] == x and block["y"] == y:
                return True
        return False


class Game:
    def __init__(self):
        pygame.font.init()
        pygame.init()
        # Create the screen
        self._screen = pygame.display.set_mode((WIDTH * SIZE_IMG, HEIGHT * SIZE_IMG))
        # Title, Icon and Settings
        self._score_font = pygame.font.SysFont("arial", 20)
        self.food, self.snake = None, None
        self.score = 0
        self.steps = 0
        self.game_over = False

    def check_collision(self):
        # Check head position
        return self.snake.head["x"] < 0 or self.snake.head["x"] >= WIDTH \
                or self.snake.head["y"] < 0 or self.snake.head["y"] >= HEIGHT \
                or self.snake.body[0] in self.snake.body[1:]


    def get_state(self):
        danger_left, danger_straight, danger_right = 0, 0, 0
        direction_left, direction_up, direction_right, direction_down = 0, 0, 0, 0
        food_left, food_up, food_right, food_down = 0, 0, 0, 0
        vector = self.snake.vector
        head_x, head_y = self.snake.head["x"], self.snake.head["y"]
        w, h = WIDTH, HEIGHT
        food_x, food_y = self.food.x, self.food.y
        body = [(b["x"], b["y"]) for b in self.snake.body[1:]]
        if vector == "left":
            direction_left = 1
            if head_y + 1 >= h or (head_x, head_y + 1) in body:
                danger_left = 1
            if head_x - 1 <= 0 or (head_x - 1, head_y) in body:
                danger_straight = 1
            if head_y - 1 <= 0 or (head_x, head_y - 1) in body:
                danger_right = 1
        if vector == "up":
            direction_up = 1
            if head_x - 1 <= 0 or (head_x - 1, head_y) in body:
                danger_left = 1
            if head_y - 1 <= 0 or (head_x, head_y - 1) in body:
                danger_straight = 1
            if head_x + 1 >= w or (head_x + 1, head_y) in body:
                danger_right = 1
        if vector == "right":
            direction_right = 1
            if head_y - 1 <= 0 or (head_x, head_y - 1) in body:
                danger_left = 1
            if head_x + 1 >= w or (head_x + 1, head_y) in body:
                danger_straight = 1
            if head_y + 1 >= h or (head_x, head_y + 1) in body:
                danger_right = 1
        if vector == "down":
            direction_down = 1
            if head_x + 1 >= w or (head_x + 1, head_y) in body:
                danger_left = 1
            if head_y + 1 >= h or (head_x, head_y + 1) in body:
                danger_straight = 1
            if head_x - 1 <= 0 or (head_x - 1, head_y) in body:
                danger_right = 1

        if food_x < head_x:
            food_left = 1
        if food_y < head_y:
            food_up = 1
        if food_x > head_x:
            food_right = 1
        if food_y > head_y:
            food_down = 1

        return [danger_left, danger_straight, danger_right,
                direction_left, direction_up, direction_right, direction_down,
                food_left, food_up, food_right, food_down]

    def reset(self):
        self.score = 0
        self.game_over = False
        self.steps = 0

    def repaint(self):
        self._screen.fill(BLACK)
        text = self._score_font.render("Score: " + str(self.score), True, ORANGE)
        self._screen.blit(text, [0, 0])
        text = self._score_font.render("Steps: " + str(self.steps), True, ORANGE)
        self._screen.blit(text, [150, 0])
        text = self._score_font.render("Coord: " + str(self.snake.head), True, ORANGE)
        self._screen.blit(text, [0, 25])
        self.food.draw()
        self.snake.draw()
        pygame.display.update()

    def step(self, action):
        last_vec = self.snake.vector
        reward = 0
        self.steps += 1
        if self.steps >= 300:
            self.game_over = True
            reward = 0
        self.snake.vector = get_vector_from_turn(action, last_vec)
        last_x, last_y = self.snake.move()
        self.repaint()
        if self.check_collision():
            self.game_over = True
            reward = -10
        if self.food.check(**self.snake.head):
            self.snake.grow(last_x, last_y)
            x, y = gen_x_y()
            while self.snake.check_pos(x, y):
                x, y = gen_x_y()
            self.food.regen(x, y)
            self.score += 1
            reward = 10
            self.steps = 0
        return reward

    def start_game(self, caption):
        pygame.display.set_caption(caption)
        self.reset()
        x, y = gen_x_y()
        self.snake = Snake(self._screen,
                           x,
                           y,
                           GREEN,
                           SIZE_IMG,
                           VECTORS[randint(0, 3)])
        while self.snake.check_pos(x, y):
            x, y = gen_x_y()
        self.food = Food(self._screen, x, y, RED, SIZE_IMG)

    def wait(self):
        pygame.time.wait(100)

    def stop_game(self):
        pygame.quit()
        return self.score


if __name__ == "__main__":
    game = Game()
    game.start_game("The Snake Game 1.0")
    game.wait()
    for i in range(10):
        game.step(choice(['left', 'straight', 'right']))
        game.wait()
    game.stop_game()
