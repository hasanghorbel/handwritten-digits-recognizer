import numpy as np

import pygame


class Block:
    def __init__(self, center: np.ndarray) -> None:
        self.center = center
        self.activation = 0

    def distance_to(self, other: np.ndarray):
        return np.linalg.norm(self.center - other)

    def activate(self, other: np.ndarray):
        distance = self.distance_to(other)

        if distance > 40:
            return

        self.activation = max(self.activation, self.f(distance))

    def f(self, distance: float):
        # return round(- (distance * 1/30) ** 4 + 1, 2)
        return- (distance * 1/30) ** 4 + 1


class Writer:
    def __init__(self, block_count) -> None:
        self.block_count = block_count
        self.screen = self.get_screen()
        self.blocks: list[Block] = self.get_blocks()

    def get_screen(self) -> pygame.Surface:
        """return square `Surface` to write on.\n
        dimensions: `(screen_width / 3, screen_width / 3)`"""

        pygame.init()

        # screen width in pixels
        screen_width = pygame.display.Info().current_w

        # square window thus height not needed
        window_dim = screen_width // 3  # third of screen width

        self.block_dim = window_dim // self.block_count
        # to ensure window fits whole blocks
        window_dim = self.block_dim * self.block_count

        return pygame.display.set_mode((window_dim, window_dim))

    def get_blocks(self):
        blocks = []
        for i in range(self.block_count ** 2):
            row, col = divmod(i, self.block_count)

            blocks.append(
                Block(
                    np.array(
                        [(col + .5) * self.block_dim, (row + .5) * self.block_dim]
                    )
                )
            )

        return blocks

    def get_block_at(self, position: np.ndarray):
        x = position[0]
        y = position[1]

        col = x // self.block_dim
        row = y // self.block_dim

        return self.blocks[row * self.block_count + col]

    def draw_blocks(self):

        for (idx, block) in enumerate(self.blocks):

            if block.activation == 0:
                continue

            col = idx % self.block_count
            row = idx // self.block_count

            x = col * self.block_dim
            y = row * self.block_dim

            color = (int(block.activation * 255), ) * 3
            pygame.draw.rect(self.screen, color,
                             (x, y, self.block_dim, self.block_dim))

    def activate_blocks(self, mouse_pos: np.ndarray):
        for block in self.blocks:
            block.activate(mouse_pos)

    def draw(self):

        clock = pygame.time.Clock()

        mouse_pressed = False
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()

                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pressed = True
                    mouse_pos = np.array(pygame.mouse.get_pos())

                    self.activate_blocks(np.array(mouse_pos))

                elif ev.type == pygame.MOUSEBUTTONUP:
                    mouse_pressed = False
                    pygame.quit()
                    return self.forward()

                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_RETURN:
                    pygame.quit()

                elif ev.type == pygame.MOUSEMOTION and mouse_pressed:
                    mouse_pos = pygame.mouse.get_pos()
                    self.activate_blocks(mouse_pos)

            self.draw_blocks()
            pygame.display.update()
            clock.tick(200)

    def forward(self) -> np.ndarray:
        return np.array(list(map(lambda x: x.activation, self.blocks)))
