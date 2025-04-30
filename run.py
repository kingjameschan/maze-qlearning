# run.py

from maze_env import MazeEnv
from q_learning import q_learning
import numpy as np

if __name__ == "__main__":
    env = MazeEnv(size=10)  # 也可同时改大迷宫

    # 训练 200 轮，每 50 轮展示一次策略
    q_table = q_learning(env,
                      episodes=8000,
                      visualize_every=80,   # 每 20 轮弹一次窗口
                      pause_time=0.1)

    # 最终一次展示
    state = env.reset()
    path = [state]
    done = False
    while not done:
        state_key = str(state)
        action = int(np.argmax(q_table[state_key]))
        state, _, done = env.step(action)
        path.append(state)
    print("Final path:", path)
    env.visualize_path(path, pause_time=0.5)
