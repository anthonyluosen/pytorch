#gym库的基本使用操作如下
import gym

# 超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
env = gym.make('CartPole-v0').unwrapped         # 使用gym库中的环境：CartPole，且打开封装

N_ACTIONS = env.action_space.n                  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]       # 杆子状态个数 (4个)
print(N_ACTIONS,N_STATES)
print(env.x_threshold,env.x_threshold)
print(env.theta_threshold_radians , env.theta_threshold_radians )