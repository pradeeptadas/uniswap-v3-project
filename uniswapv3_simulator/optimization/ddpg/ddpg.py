import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging

from .replay_buffer import ReplayBuffer
from .exploration_noise import ConstantNoise


logger = logging.getLogger('optimization.ddpg')


class DDPG:
    def __init__(self,
                 model=None,
                 gamma=0.99,
                 tau=1e-3,
                 optimizer=optim.Adam,
                 actor_optimizer_kwargs={},
                 critic_optimizer_kwargs={},
                 clip_gradients=None):

        self.online_network = model
        self.target_network = copy.deepcopy(model)
        self.online_network.eval()
        self.target_network.eval()

        self.actor_optimizer = optimizer(self.online_network.actor_params,
                                         **actor_optimizer_kwargs)
        self.critic_optimizer = optimizer(self.online_network.critic_params,
                                          **critic_optimizer_kwargs)

        self.gamma = gamma
        self.tau = tau
        self.clip_gradients = clip_gradients

    def action(self, obs):
        self.online_network.eval()
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.online_network.action(obs).cpu().numpy()

        return action

    def update_target_networks(self):
        for target, online in zip(self.target_network.parameters(),
                                  self.online_network.parameters()):
            target.detach_()
            target.copy_(target * (1.0 - self.tau) + online * self.tau)

        # this is for things like batch norm and other PyTorch objects
        # that have buffers and/or instead of learnable parameters
        for target, online in zip(self.target_network.buffers(),
                                  self.online_network.buffers()):
            # detach is probably unnecessary since buffers are not learnable
            target.detach_()
            target.copy_(target * (1.0 - self.tau) + online * self.tau)

    def update_target(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            next_action = self.target_network.action(next_obs)
            q_sa_next = self.target_network.critic_value(next_obs, next_action)

        update_target = reward + self.gamma * q_sa_next * ~terminal

        return update_target

    def update(self, obs, action, reward, next_obs, terminal):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
        terminal = torch.as_tensor(terminal, dtype=torch.bool)

        self.online_network.eval()
        update_target = self.update_target(obs, action, reward, next_obs, terminal)

        self.online_network.train()
        q_sa = self.online_network.critic_value(obs, action)
        td_error = q_sa - update_target
        critic_loss = td_error.pow(2).mul(0.5).squeeze(-1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_gradients:
            nn.utils.clip_grad_norm_(self.online_network.critic_params,
                                     self.clip_gradients)
        self.critic_optimizer.step()

        action = self.online_network.action(obs)
        policy_loss = -self.online_network.critic_value(obs, action).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.clip_gradients:
            nn.utils.clip_grad_norm_(self.online_network.actor_params,
                                     self.clip_gradients)
        self.actor_optimizer.step()

        self.update_target_networks()
        self.online_network.eval()


class LinearActorModel(nn.Module):
    def __init__(self, obs_size, action_size,
                 critic_hidden_layers):
        super().__init__()

        critic_layers = []
        input_size = obs_size + action_size
        for i, units in enumerate(critic_hidden_layers):
            output_size = units
            critic_layers.append(nn.Linear(input_size, output_size))
            critic_layers.append(nn.ReLU())
            input_size = output_size

        critic_layers.append(nn.Linear(input_size, 1))
        self.critic_layers = nn.Sequential(*critic_layers)
        self.critic_params = list(self.critic_layers.parameters())

        self.actor_layers = nn.Linear(obs_size, action_size)
        self.actor_params = list(self.actor_layers.parameters())

    def forward(self, obs):
        action = self.actor_layers(obs)
        # using sigmoid instead of tanh since "actions" are probably >= 0?
        action = torch.sigmoid(action).squeeze()

        return action

    def action(self, obs):
        return self.forward(obs)

    def critic_value(self, obs, action):
        x = torch.cat([obs, action], dim=-1)

        return self.critic_layers(x)


class DeepActorModel(nn.Module):
    def __init__(self, obs_size, action_size,
                 actor_hidden_layers, critic_hidden_layers):
        super().__init__()

        critic_layers = []
        input_size = obs_size + action_size
        # critic_layers.append(nn.BatchNorm1d(input_size))
        for i, units in enumerate(critic_hidden_layers):
            output_size = units
            critic_layers.append(nn.Linear(input_size, output_size))
            critic_layers.append(nn.ReLU())
            input_size = output_size

        critic_layers.append(nn.Linear(input_size, 1))
        self.critic_layers = nn.Sequential(*critic_layers)
        self.critic_params = list(self.critic_layers.parameters())

        actor_layers = []
        input_size = obs_size
        # actor_layers.append(nn.BatchNorm1d(input_size))
        for i, units in enumerate(actor_hidden_layers):
            output_size = units
            actor_layers.append(nn.Linear(input_size, output_size))
            actor_layers.append(nn.ReLU())
            input_size = output_size

        actor_layers.append(nn.Linear(input_size, action_size))
        self.actor_layers = nn.Sequential(*actor_layers)
        self.actor_params = list(self.actor_layers.parameters())

    def forward(self, obs):
        action = self.actor_layers(obs)
        # using sigmoid instead of tanh since "actions" are probably >= 0?
        action = torch.sigmoid(action).squeeze()

        return action

    def action(self, obs):
        return self.forward(obs)

    def critic_value(self, obs, action):
        x = torch.cat([obs, action], dim=-1)

        return self.critic_layers(x)


class DDPGTrainer:
    def __init__(self, agent, env, args):
        self.agent = agent
        self.env = env
        self.args = args

        self._replay_buffer = None
        self._exploration_noise = None

    def train(self):
        seed_seq = np.random.SeedSequence(self.args.seed)
        seeds = seed_seq.generate_state(3)

        self.env.seed(int(seeds[0]))
        self._replay_buffer = ReplayBuffer(self.args.memory_size, seed=seeds[1])
        self._exploration_noise = self.args.exploration_noise(
            **self.args.noise_kwargs,
            seed=seeds[2]
        )
        rewards = []

        obs = self.env.reset()
        total_reward = 0
        j = 0
        try:
            for i in range(self.args.train_steps):
                action = self.agent.action(obs.reshape(1, -1)).squeeze()
                logger.debug(f'Raw action: {action}')
                exploration_noise = self._exploration_noise.sample()
                logger.debug(f'Exploration noise: {exploration_noise}')
                action += exploration_noise
                if self.args.clip_actions:
                    action = np.clip(action, self.args.clip_actions[0], self.args.clip_actions[1])
                logger.debug(f'Final action: {action}')

                next_obs, reward, terminal, _ = self.env.step(action)
                self._replay_buffer.add((obs, action, reward, next_obs, terminal))
                total_reward += reward

                if (i >= self.args.update_start) and (i % self.args.update_freq == 0):
                    exp_batch = self._replay_buffer.sample(self.args.batch_size)
                    exp_batch = list(zip(*exp_batch))
                    self.agent.update(
                        np.array(exp_batch[0]),  # obs
                        np.array(exp_batch[1]),  # action
                        np.array(exp_batch[2]).reshape(-1, 1),  # reward
                        np.array(exp_batch[3]),  # next_obs
                        np.array(exp_batch[4]).reshape(-1, 1),  # terminal
                    )

                obs = next_obs
                j += 1
                if terminal:
                    rewards.append(total_reward)
                    if len(rewards) % 50 == 0:
                        ep_num = len(rewards)
                        mean_score = np.mean(rewards[-50:])
                        print(
                            f'Episode: {ep_num:>5,} | Time Steps: {j:>5,} | '
                            f'Mean Score: {mean_score:>7,.2f}'
                        )
                    obs = self.env.reset()
                    total_reward = 0
                    j = 0

        except KeyboardInterrupt:
            logger.warning(f'Training stopped during step {i:,.0f}.')

        self.env.close()

        return rewards


class TrainArgs:
    def __init__(self, **kwargs):
        self.train_steps = 10000
        self.batch_size = 64
        self.update_freq = 1
        self.update_start = 1000
        self.memory_size = 1000000
        self.exploration_noise = ConstantNoise
        self.noise_kwargs = {}
        self.clip_actions = None
        self.seed = None

        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                logger.warning(f'{key} is not a valid training parameter.')
                continue
            setattr(self, key, value)