import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import time
import gym


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class DQN(nn.Module):
    '''
    DQN类
    '''

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 seed,
                 layer_type="ff"):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size

        self.head_1 = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        weight_init([self.head_1, self.ff_1])

    def forward(self, input):
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out


class ReplayBuffer:
    """固定大小的回放区来存储经验"""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        """
        初始化 ReplayBuffer.
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

    def add(self, state, action, reward, next_state, done):
        """添加经验."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(
            )
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * self.n_step_buffer[idx][2]

        return self.n_step_buffer[0][0], self.n_step_buffer[0][
            1], Return, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

    def sample(self):
        """从memory中随机抽取经验."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state for e in experiences
                      if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences
                      if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                       ]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """返回当前memory大小."""
        return len(self.memory)


class Munchausen_DQN():

    def __init__(self, state_size, action_size, layer_size, BATCH_SIZE,
                 BUFFER_SIZE, LR, TAU, GAMMA, UPDATE_EVERY, device, seed):
        """
        初始化智能体
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.action_step = 4
        self.last_action = None

        # Q网络
        self.qnetwork_local = DQN(state_size, action_size, layer_size,
                                  seed).to(device)
        self.qnetwork_target = DQN(state_size, action_size, layer_size,
                                   seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)

        # 回放缓冲区
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed,
                                   self.GAMMA, 1)

        # 初始化 time step
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, writer):
        # 保存经验在 replay memory
        self.memory.add(state, action, reward, next_state, done)

        # 学习
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # 如果内存中有足够的样本可用，则获取随机子集并学习
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0.):
        """
        根据当前策略返回给定状态的操作。 每4帧执行一次
        """

        if self.action_step == 4:
            state = np.array(state)

            state = torch.from_numpy(state).float().unsqueeze(0).to(
                self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy
            if random.random() > eps:  # 如果随机数大于 epsilon 选择贪心操作
                action = np.argmax(action_values.cpu().data.numpy())
                self.last_action = action
                return action
            else:
                action = random.choice(np.arange(self.action_size))
                self.last_action = action
                return action
        else:
            self.action_step += 1
            return self.last_action

    def learn(self, experiences):
        """
        使用给定的经验元组更新价值参数
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # 从 target model 得到下回合预测Q值
        Q_targets_next = self.qnetwork_target(next_states).detach()
        # 用logsum计算熵项
        logsum = torch.logsumexp(\
                                (Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1))/entropy_tau , 1).unsqueeze(-1)

        tau_log_pi_next = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(
            -1) - entropy_tau * logsum
        # 目标策略
        pi_target = F.softmax(Q_targets_next / entropy_tau, dim=1)
        Q_target = (self.GAMMA *
                    (pi_target * (Q_targets_next - tau_log_pi_next) *
                     (1 - dones)).sum(1)).unsqueeze(-1)

        # 用 logsum技巧计算munchausen addon
        q_k_targets = self.qnetwork_target(states).detach()
        v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
        logsum = torch.logsumexp((q_k_targets - v_k_target) / entropy_tau,
                                 1).unsqueeze(-1)
        log_pi = q_k_targets - v_k_target - entropy_tau * logsum
        munchausen_addon = log_pi.gather(1, actions)

        # 计算munchausen奖励:
        munchausen_reward = (
            rewards + alpha * torch.clamp(munchausen_addon, min=lo, max=0))

        # 计算当前状态Q target
        Q_targets = munchausen_reward + Q_target

        # 从local model获得Q_expected
        q_k = self.qnetwork_local(states)
        Q_expected = q_k.gather(1, actions)

        # 计算损失
        loss = F.mse_loss(Q_expected, Q_targets)  #mse_loss
        # 最小化损失
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        """Soft 更新 model 参数
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data +
                                    (1.0 - self.TAU) * target_param.data)


def eval_runs(eps, frame):
    """
    使用当前 epsilon 进行评估运行
    """
    env = gym.make("CartPole-v0")
    reward_batch = []
    for i in range(5):
        state = env.reset()
        rewards = 0
        while True:
            action = agent.act(state, eps)
            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)

    writer.add_scalar("Reward", np.mean(reward_batch), frame)


def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01):
    """
    深度 Munchausen Q学习.
    """
    scores = []  # 每回合分数
    scores_window = deque(maxlen=100)  # last 100 scores
    output_history = []
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    i_episode = 1
    state = env.reset()
    score = 0
    for frame in range(1, frames + 1):

        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done, writer)
        state = next_state
        score += reward
        # 线性退火到最小 epsilon 值直到 eps_frames，然后慢慢将 epsilon 降低到 0 直到训练结束
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame * (1 / eps_frames)), min_eps)
            else:
                eps = max(
                    min_eps - min_eps * ((frame - eps_frames) /
                                         (frames - eps_frames)), 0.001)

        # 评估运行
        if frame % 1000 == 0:
            eval_runs(eps, frame)

        if done:
            scores_window.append(score)  # 保存最近最多分数
            scores.append(score)
            writer.add_scalar("Average100", np.mean(scores_window), frame)
            output_history.append(np.mean(scores_window))
            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(
                i_episode, frame, np.mean(scores_window)),
                  end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(
                    i_episode, frame, np.mean(scores_window)))
            i_episode += 1
            state = env.reset()
            score = 0

    return output_history


if __name__ == "__main__":

    writer = SummaryWriter("runs/" + "M-DQN_CP_new_4")
    seed = 4
    BUFFER_SIZE = 100000
    BATCH_SIZE = 8
    GAMMA = 0.99
    TAU = 1e-2
    LR = 1e-3
    UPDATE_EVERY = 1
    lo = -1
    entropy_tau = 0.03
    alpha = 0.9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    np.random.seed(seed)
    env = gym.make("CartPole-v0")

    env.seed(seed)
    action_size = env.action_space.n
    state_size = env.observation_space.shape

    agent = Munchausen_DQN(state_size=state_size,
                           action_size=action_size,
                           layer_size=256,
                           BATCH_SIZE=BATCH_SIZE,
                           BUFFER_SIZE=BUFFER_SIZE,
                           LR=LR,
                           TAU=TAU,
                           GAMMA=GAMMA,
                           UPDATE_EVERY=UPDATE_EVERY,
                           device=device,
                           seed=seed)

    # 将 epsilon 帧设置为 0
    eps_fixed = False

    t0 = time.time()
    final_average100 = run(frames=45000,
                           eps_fixed=eps_fixed,
                           eps_frames=5000,
                           min_eps=0.025)
    t1 = time.time()
