                          # 导入numpy
import gym                                      # 导入gym
from DQN import DQN
# 超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
env = gym.make('CartPole-v0').unwrapped         # 使用gym库中的环境：CartPole，且打开封装


dqn = DQN()                                                             # 令dqn=DQN类

for i in range(400):                                                    # 400个episode循环
    print('<<<<<<<<<Episode: %s' % i)
    s = env.reset()                                                     # 重置环境
    episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励

    while True:                                                         # 开始一个episode (每一个循环代表一步)
        env.render()                                                    # 显示实验动画
        a = dqn.choose_action(s)
        # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)                                 # 执行动作，获得反馈
        # print(r)
        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2
        # new_r = r
        dqn.store_transition(s, a, new_r, s_)                 # 存储样本
        episode_reward_sum += new_r                           # 逐步加上一个episode内每个step的reward

        s = s_                                                # 更新状态
        # print(s)
        if dqn.memory_counter > MEMORY_CAPACITY:              # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.learn()

        if done:       # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            break                                             # 该episode结束
env.close()