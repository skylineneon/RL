import torch
from torch import nn
import gym
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
import random
import math


eps = torch.finfo(torch.float32).eps


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(32, 2),
        )

    def forward(self, state):
        f = self.net(state)
        value = self.value_net(f)
        return value


class Trainer:
    def __init__(self):
        self.agent = Agent()  # 定义一个训练agent
        self.target_agent = Agent()  # 定义一个采样agent
        self.target_agent.eval()
        self.env = gym.make('CartPole-v1')
        self.opt = torch.optim.Adam(self.agent.parameters())

        self.pool = Pool()  # 创建一个经验池
        self.dataloader = DataLoader(self.pool.getDataset(), 32, True)
        self.max_reward = 0  # 采样玩一次游戏回报之和最大值
        self.reward_num = 0  # 最大值出现的计数器
        self.frame_idx = 0  # 动作选择计数器
        self.epsilon = lambda frame_idx: 0.01 + (1 - 0.01) * math.exp(-1. * frame_idx / 200.)  # 指数减小的数
        self.gamma = 0.99  # 折扣系数

    # 主函数
    def __call__(self):
        for epoch in range(1000000):
            # 采样
            self._samlpe(epoch)
            if not self.pool.is_full(): continue  # 将经验池装满，才开始训练
            # 训练
            if epoch == 700:
                self.env = gym.make('CartPole-v1',render_mode='human')
            self._train()

    # 采样部分
    def _samlpe(self, epoch):
        # 采样
        state = self.env.reset()[0]
        rewards = 0
        while True:
            action = self.__action_select(state)
            next_state, reward, done,_, info = self.env.step(action)
            self.pool.put([action, state, reward, next_state, done])
            if done:
                break
            state = next_state
            rewards+=reward

        self.reward_num+=1
        if rewards > self.max_reward:
            self.max_reward = rewards
            self.reward_num = 0
        print(f"{epoch}：reward：{rewards}， max_reward:{self.max_reward},time:{self.reward_num}")

    # 训练部分
    def _train(self):
        for _action, _state, _reward, _next_state, _done in self.dataloader:
            q_values = self.agent(_state.float())
            q_values = torch.gather(q_values, dim=1, index=_action[:, None])[:, 0]

            q_values_next = self.target_agent(_next_state.float()).max(dim=1)[0].detach()
            q_values_target = _reward + self.gamma * q_values_next * (1 - torch.tensor(_done, dtype=torch.float32))

            loss = torch.mean((q_values - q_values_target) ** 2)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        # 给采样网络赋参数
        self.target_agent.load_state_dict(self.agent.state_dict())

    def __action_select(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            state = torch.from_numpy(state).float()
            q_values = self.target_agent(state[None])
            # 去除数据和网络的粘连
            q_values = q_values.detach()
            action = q_values.max(dim=1)[1].item()
        else:
            action = random.randrange(2)  # 从动作0 1中随机选择一个
        return action


# 经验池
class Pool:
    def __init__(self, cap=1000):
        self.cap = cap  # 经验池存放的最多数据条数
        self.datas = []  # 经验池

    def put(self, record):
        if self.is_full(): del self.datas[0]  # 如果经验池装满，则删除第一条记录
        self.datas.append(record)  # 装入经验池

    def is_full(self):
        return len(self.datas) > self.cap  # 满了：True

    def getDataset(self):  # 获取Dataset
        return PoolDataset(self.datas)


class PoolDataset(Dataset):
    def __init__(self, datas):
        super(PoolDataset, self).__init__()
        self.datas = datas

    def __len__(self):
        return 32 * 10

    def __getitem__(self, index):
        return self.datas[index]


if __name__ == '__main__':
    train = Trainer()
    train()
