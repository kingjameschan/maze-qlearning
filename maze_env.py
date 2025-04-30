# maze_env.py - 简单迷宫环境定义 + 动态可视化
import numpy as np
import pygame
import random
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib import animation

class MazeEnv:
    def __init__(self, size=5):
        random.seed(1424)
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start
        self.maze = np.zeros((size, size), dtype=int)
        self.generate_walls()

    def generate_walls(self):
        for _ in range(self.size):
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (i, j) != self.start and (i, j) != self.goal:
                self.maze[i][j] = 1

    def reset(self):
        self.state = self.start
        return self.state

    def is_terminal(self, state):
        return state == self.goal

    def step(self, action):
        i, j = self.state
        if action == 0: i -= 1
        if action == 1: i += 1
        if action == 2: j -= 1
        if action == 3: j += 1

        if 0 <= i < self.size and 0 <= j < self.size and self.maze[i][j] == 0:
            self.state = (i, j)

        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal
        return self.state, reward, done

    def render(self):
        maze_copy = self.maze.copy()
        i, j = self.state
        maze_copy[i][j] = 8
        print(maze_copy)

    def visualize_path(self, path, pause_time=0.5, tile_size=80):
            """
            用 Pygame 逐步展示智能体走迷宫的路径。
            pause_time: 每步停留秒数
            tile_size: 每个格子的像素大小
            """
            pygame.init()
            size = self.size
            screen = pygame.display.set_mode((size*tile_size, size*tile_size))
            clock = pygame.time.Clock()

            running = True
            for step, (x, y) in enumerate(path):
                # 处理退出事件
                for evt in pygame.event.get():
                    if evt.type == pygame.QUIT:
                        running = False
                if not running:
                    break

                # 1. 清屏
                screen.fill((255,255,255))

                # 2. 画障碍和空地
                for i in range(size):
                    for j in range(size):
                        color = (0,0,0) if self.maze[i][j] == 1 else (200,200,200)
                        pygame.draw.rect(screen, color,
                                        (j*tile_size, i*tile_size, tile_size, tile_size))

                # 3. 画已走路径
                for (px, py) in path[:step+1]:
                    pygame.draw.rect(screen, (150,150,255),
                                    (py*tile_size, px*tile_size, tile_size, tile_size))

                # 4. 画当前智能体（红色）
                pygame.draw.rect(screen, (255,50,50),
                                (y*tile_size, x*tile_size, tile_size, tile_size))

                # 5. 画终点（绿色）
                gx, gy = self.goal
                pygame.draw.rect(screen, (50,255,50),
                                (gy*tile_size, gx*tile_size, tile_size, tile_size))

                # 6. 刷新屏幕 & 延时
                pygame.display.flip()
                pygame.time.delay(int(pause_time * 1000))

            # 最后保持窗口直到用户关闭
            while running:
                for evt in pygame.event.get():
                    if evt.type == pygame.QUIT:
                        running = False
                clock.tick(10)

            pygame.quit()