import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import namedtuple, deque
from env_creator import qsimpy_env_creator

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, output_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

def vtrace_advantages(behaviour_log_probs, target_log_probs, rewards, values, bootstrap_value, gamma=0.99, rho_bar=1.0, c_bar=1.0):
    deltas = rewards + gamma * bootstrap_value - values
    rho = torch.exp(target_log_probs - behaviour_log_probs)
    rho_clipped = torch.clamp(rho, max=rho_bar)
    
    advantages = []
    gae = 0.0
    for delta, r_clipped in zip(reversed(deltas), reversed(rho_clipped)):
        gae = delta + gamma * r_clipped * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages, dtype=torch.float32)

def train_impala(env_name="CartPole-v1", num_episodes=1000, gamma=0.99, lr=1e-3):
    env_config={
                "obs_filter": "rescale_-1_1",
                "reward_filter": None,
                "dataset": "qdataset/qsimpyds_1000_sub_36.csv",
            }
    env = qsimpy_env_creator(env_config)
    # env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = ActorCritic(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        log_probs = []
        values = []
        rewards = []

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            policy_logits, value = policy(obs_tensor)
            
            action_prob = torch.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            obs, reward, done, _, _ = env.step(action.item())

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            episode_rewards.append(reward)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        rewards = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            bootstrap_value = values[-1] if not done else torch.tensor(0.0)
        # Ensure rewards length matches log_probs and values
        rewards = rewards[:-1]
        advantages = vtrace_advantages(log_probs[:-1], log_probs[1:], rewards, values[:-1], bootstrap_value, gamma)

        value_loss = ((values[:-1] - advantages) ** 2).mean()
        policy_loss = -(advantages.detach() * log_probs[:-1]).mean()

        loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode}: Reward = {sum(episode_rewards)}")

if __name__ == "__main__":
    train_impala()
