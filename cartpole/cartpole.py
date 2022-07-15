import math
import random
from typing import Text, Dict, List, NamedTuple, Tuple, Optional, Union
from collections import namedtuple
from itertools import count
from PIL import Image
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T 
from IPython import display
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
class DQN(nn.Module):
    def __init__(self, num_state_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_state_features,
                             out_features=32)
        self.fc2 = nn.Linear(in_features=32,
                             out_features=64)
        self.fc3 = nn.Linear(in_features=64,
                             out_features=128)
        self.out = nn.Linear(in_features=128,
                             out_features=2) 
    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)
        return t
        
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)
e = Experience(2, 3, 1, 4)

class ReplayMemeory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    def push(self, experience) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    def can_provide_sample(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    def get_exploration_rate(self, current_step: int) -> float:
        return self.end + (self.start - self.end) *\
                math.exp(-1. * current_step * self.decay)

class Agent:
    def __init__(self, strategy, num_actions, device) -> None:
        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0
        self.device = device
    def select_action(self, state, policy_net) -> float:
        epsilon_rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if epsilon_rate > random.random(): 
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(device=self.device) 
        else:
            with torch.no_grad(): 
                return policy_net(state).\
                       unsqueeze(dim=0).\
                       argmax(dim=1).\
                       to(device=self.device) 

class CartPoleEnvManager:
    def __init__(self, device) -> None:
        self.device = device
        self.env = gym.make('CartPole-v1').unwrapped 
        self.env.reset() 
        self.current_screen = None 
        self.current_state = None
        self.done = False
    def reset(self) -> None:
        self.current_state = self.env.reset()
    def close(self) -> None:
        self.env.close()
    def render(self, mode='human'):
        return self.env.render(mode)
    def display(self):
        return self.env.display()
    def num_actions_available(self):
        return self.env.action_space.n
    def take_action(self, action: torch.Tensor) -> torch.Tensor:
        self.current_state, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)
    def get_state(self):
        if self.done:
            return torch.zeros_like(
              torch.tensor(self.current_state, device=self.device)
            ).float()
        else:
            return torch.tensor(self.current_state, device=self.device).float()
    def num_state_features(self):
        return self.env.observation_space.shape[0]
    def image_reset(self) -> None:
        self.env.reset()
        self.current_screen = None 
    def get_image_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    def just_starting(self) -> bool:
        return self.current_screen is None
    def get_screen_height(self) -> int:
        screen = self.get_processed_screen()
        return screen.shape[2]
    def get_screen_width(self) -> int:
        screen = self.get_processed_screen()
        return screen.shape[3]
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen
    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, 
                                      dtype=np.float32) / 255 
        screen = torch.from_numpy(screen)
        resize = T.Compose([
                           T.ToPILImage(), 
                           T.Resize((40,90)), 
                           T.ToTensor() 
        ])
        return resize(screen).unsqueeze(0).to(self.device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_em = CartPoleEnvManager(device)
image_em.image_reset()
screen = image_em.render('rgb_array')

plt.figure()
plt.imshow(screen)
plt.title('None-processed Screen Example')
plt.show()

screen = image_em.get_processed_screen()
plt.figure()
plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(), interpolation='none')
plt.title('Processed Screen Example')
plt.show()

screen = image_em.get_image_state()
plt.figure()
plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(), interpolation='none')
plt.title('Example of starting state')
plt.show()

for i in range(5):
    image_em.take_action(torch.tensor([1]))
screen = image_em.get_image_state()
plt.figure()
plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(), interpolation='none')
plt.title('Example of non-starting state')
plt.show()

image_em.done = True
screen = image_em.get_image_state()
plt.figure()
plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(), interpolation='none')
plt.title('Example of End state')
plt.show()

def extract_tensors(experiences: NamedTuple) -> Tuple[torch.TensorType]:
    batch = Experience(*zip(*experiences))
    t_states = torch.stack(batch.state)
    t_actions = torch.cat(batch.action)
    t_next_state = torch.stack(batch.next_state)
    t_rewards = torch.cat(batch.reward)
    return (t_states,
            t_actions,
            t_next_state,
            t_rewards)

e1 = Experience(1,1,1,1)
e2 = Experience(2,2,2,2)
e3 = Experience(3,3,3,3)
experiences = [e1,e2,e3]
experiences
batch = Experience(*zip(*experiences))
batch

class QValues:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    @staticmethod
    def get_next(target_net, next_states):
        final_states_location = next_states.flatten(start_dim=1)\
          .max(dim=1)[0].eq(0).type(torch.bool) 
        non_final_states_locations = (final_states_location == False)
        non_final_states = next_states[non_final_states_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_states_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

def plot(values, # Episode Durations
         moving_avg_period,
         env=None): # 100 episodes moving average
    moving_avg = get_moving_average(moving_avg_period, values)
    # plt.figure(2)
    figure, axes = plt.subplots(1, 2, figsize=(21, 6),
                              gridspec_kw={'width_ratios': [4, 6],
                                           'wspace':0.025, 'hspace':0.025})
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.clf()
    axes[0].set_title('Training...')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Duration')
    axes[0].plot(values)
    axes[0].plot(moving_avg)
    if env:
        axes[1].set_title('Cart Pole Rendering')
        axes[1].grid(False)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].imshow(env.render('rgb_array'))
    else:
        axes[1].set_axis_off()
    plt.pause(0.001)
    print(f"- Episodes: {len(values)}\n- {moving_avg_period} episodes moving avg: {moving_avg[-1]}")
    if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period: 
        moving_avg = values.unfold(dimension=0, 
                                   size=period, 
                                   step=1)\
                                   .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()  

plot(np.random.rand(300), 100)
batch_size = 256
gamma = 0.999 
eps_start = 1 
eps_end = 0.01 
eps_decay = 0.001 
target_update = 10 
memory_size = 100000
lr = 0.001
num_episodes = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemeory(memory_size)
policy_net = DQN(em.num_state_features()).to(device)
target_net = DQN(em.num_state_features()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
episode_duration = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()
    for timestep in count():
        em.render()
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_states = em.get_state()
        memory.push(Experience(state, action, next_states, reward))
        state = next_states
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, next_states, rewards = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = rewards + (gamma * next_q_values)
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if em.done:
            episode_duration.append(timestep)
            #plot(episode_duration, 100, em) 
            break
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if get_moving_average(100, episode_duration)[-1] >= 195:
        break
em.close()