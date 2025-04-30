# q_learning.py

import numpy as np
import random

def q_learning(env,
               episodes=1000,
               alpha=0.1,
               gamma=0.9,
               epsilon=0.1,
               visualize_every=None,   # 新增：每隔多少集可视化一次
               pause_time=0.2):        # 新增：可视化时每步暂停秒数
    """
    返回 q_table。
    如果 visualize_every 不为 None，就在每隔 N 集后调用 env.visualize_path() 展示一次当时策略。
    """
    q_table = dict()
    actions = [0, 1, 2, 3]

    for ep in range(1, episodes+1):
        state = env.reset()
        state_key = str(state)
        q_table.setdefault(state_key, [0]*4)

        done = False
        step_count = 0
        max_steps = env.size * env.size * 2

        while not done and step_count < max_steps:
            step_count += 1
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = int(np.argmax(q_table[state_key]))

            next_state, reward, done = env.step(action)
            next_key = str(next_state)
            q_table.setdefault(next_key, [0]*4)

            # Q-update
            best_next = max(q_table[next_key])
            q_table[state_key][action] += alpha * (reward + gamma*best_next - q_table[state_key][action])
            state_key = next_key

        # 打印进度
        if ep % max(1, episodes//10) == 0:
            print(f"Episode {ep}/{episodes} finished.")

        # 可视化当前策略
        if visualize_every and ep % visualize_every == 0:
            # 根据当前q_table生成路径
            path = []
            s = env.reset()
            p_done = False
            path.append(s)
            while not p_done:
                key = str(s)
                a = int(np.argmax(q_table[key]))
                s, _, p_done = env.step(a)
                path.append(s)
                if len(path) > env.size*env.size*5:  # 超过 5×地图大小 则 break
                    print("▶️ 路径生成挂死了，跳出")
                    break
            print(f"Visualizing policy after episode {ep} ...")
            env.visualize_path(path, pause_time=pause_time)

    return q_table
