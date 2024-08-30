import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define the Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        logits = self.fc(state)
        return Categorical(logits=logits)

# Define the Value Network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.fc(state)

# PPO Algorithm
class PPO:
    def __init__(self, state_dim, action_dim, policy_lr=3e-4, value_lr=1e-3, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        dist = self.policy(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def update(self, trajectories):
        states = torch.tensor(trajectories['states'], dtype=torch.float32)
        actions = torch.tensor(trajectories['actions'], dtype=torch.int64)
        old_log_probs = torch.tensor(trajectories['log_probs'], dtype=torch.float32)
        returns = torch.tensor(trajectories['returns'], dtype=torch.float32)
        
        for _ in range(self.k_epochs):
            # Compute new log probs and state values
            dist = self.policy(states)
            new_log_probs = dist.log_prob(actions)
            state_values = self.value(states).squeeze()
            
            # Compute advantages
            advantages = returns - state_values.detach()
            
            # Compute ratio of new and old probabilities
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(state_values, returns)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Update value function
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

# Training loop
def train():
    env = gym.make('CartPole-v1')
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    
    max_episodes = 1000
    max_timesteps = 300
    update_timestep = 2000
    trajectory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'returns': []}
    
    timestep = 0
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            
            state = next_state
            episode_reward += reward
            timestep += 1
            
            if timestep % update_timestep == 0:
                next_value = ppo.value(torch.tensor(next_state, dtype=torch.float32)).item()
                returns = ppo.compute_returns(trajectory['rewards'], trajectory['dones'], next_value)
                trajectory['returns'] = returns
                
                ppo.update(trajectory)
                
                # Clear trajectory
                trajectory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'returns': []}
                timestep = 0
            
            if done:
                break
        
        print(f"Episode {episode+1} Reward: {episode_reward}")

if __name__ == "__main__":
    train()
