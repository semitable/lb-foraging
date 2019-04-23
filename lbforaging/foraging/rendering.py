import pygame
import os

# Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)

script_dir = os.path.dirname(__file__)

img_apple = pygame.image.load(os.path.join(script_dir, "icons/apple.png"))
img_agent = pygame.image.load(os.path.join(script_dir, "icons/agent.png"))

class Viewer:
    def __init__(self, world_size):

        self.rows, self.cols = world_size

        self.grid_size = 50
        self.icon_size = 20

        self.name_font_size = 10
        self.level_font_size = 20
        pygame.init()
        self._screen = pygame.display.set_mode(
            (self.cols * self.grid_size + 1, self.rows * self.grid_size + 1)
        )
        self._name_font = pygame.font.SysFont("monospace", self.name_font_size)
        self._level_font = pygame.font.SysFont("monospace", self.level_font_size)

        self.img_apple = pygame.transform.scale(
            img_apple, (self.icon_size, self.icon_size)
        )
        self.img_agent = pygame.transform.scale(
            img_agent, (self.icon_size, self.icon_size)
        )

    def render(self, env):

        self._screen.fill(_BLACK)
        self._draw_grid()
        self._draw_food(env)
        self._draw_players(env)

        pygame.display.flip()

    def _draw_grid(self):
        for r in range(self.rows + 1):
            pygame.draw.line(
                self._screen,
                _WHITE,
                (0, self.grid_size * r),
                (self.grid_size * self.cols, self.grid_size * r),
            )
        for c in range(self.cols + 1):
            pygame.draw.line(
                self._screen,
                _WHITE,
                (self.grid_size * c, 0),
                (self.grid_size * c, self.grid_size * self.rows),
            )

    def _draw_food(self, env):
        for r in range(self.rows):
            for c in range(self.cols):
                if env.field[r, c] != 0:
                    self._draw_population_in_cell(
                        self.img_apple,
                        (self.grid_size * c, self.grid_size * r),
                        env.field[r, c],
                    )

    def _draw_population_in_cell(self, img, location, number):
        offset = 5
        coords = [
            (0, 0),
            (0, self.icon_size),
            (self.icon_size, 0),
            (self.icon_size, self.icon_size),
        ]

        for i in coords[:number]:
            self._screen.blit(
                img, (location[0] + i[0] + offset, location[1] + i[1] + offset)
            )

    def _draw_players(self, env):
        for player in env.players:
            r, c = player.position
            self._draw_population_in_cell(
                self.img_agent, (self.grid_size * c, self.grid_size * r), player.level
            )
            self._screen.blit(
                self._name_font.render(str(player.name), 1, _WHITE),
                (
                    self.grid_size * c + self.grid_size // 3 - 5,
                    self.grid_size * r + self.grid_size // 3 + 20,
                ),
            )
