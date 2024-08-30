import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),  # Increased number of neurons
            nn.ReLU(),
            nn.Linear(256, 256),  # Added another layer
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return torch.softmax(self.fc(state.unsqueeze(0)), dim=1)
    
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),  # Increased number of neurons
            nn.ReLU(),
            nn.Linear(256, 256),  # Added another layer
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.fc(state)

def update_pol(rs, ss, nss, pol, pol_opt, val_net, a_s, gam):
    discounted_rewards = []
    running_add = 0
    for r in rs[::-1]:
        running_add = r + gam * running_add
        discounted_rewards.insert(0, running_add)
    
    discounted_rewards = torch.tensor(discounted_rewards)
    values = val_net(ss).squeeze()
    next_values = val_net(nss).squeeze()
    advantages = discounted_rewards - values

    pol_loss = 0  # Initialize the policy loss
    for log_prob, advantage in zip(a_s, advantages):
        pol_loss += -log_prob * advantage  # Accumulate losses

    pol_opt.zero_grad()
    pol_loss.backward()  # Single backward pass
    pol_opt.step()

def update_val(rs, ss, ns, crit, crit_opt, gam=0.99):
    discounted_rewards = []
    running_add = 0
    for r in rs[::-1]:
        running_add = r + gam * running_add
        discounted_rewards.insert(0, running_add)
    
    discounted_rewards = torch.tensor(discounted_rewards)
    values = crit(ss).squeeze()
    value_loss = F.mse_loss(values, discounted_rewards)
    
    crit_opt.zero_grad()
    value_loss.backward()
    crit_opt.step()

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNet(state_dim, action_dim)
value_net = ValueNet(state_dim)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-4)

best_reward = 0

for episode in range(1000):
    state = env.reset()
    # state = torch.tensor(state, dtype=torch.float32)
    state = torch.tensor(state[0], dtype=torch.float32) if len(state)==2 else torch.tensor(state, dtype=torch.float32) 
    total_reward = 0
    rewards = []
    states = []
    next_states = []
    log_probs = []
    actions = []

    for t in range(200):
        action_probs = policy_net(state)
        action = torch.multinomial(action_probs, 1).item()
        # next_state, reward, done, _ = env.step(action)
        next_state, reward, done, _, _ = env.step(action)

        next_state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward
        
        log_prob = torch.log(action_probs[0][action])
        
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        log_probs.append(log_prob)
        actions.append(action)

        state = next_state

        if done:
            break

    states = torch.stack(states)
    next_states = torch.stack(next_states)

    update_val(rewards, states, next_states, value_net, value_optimizer)
    update_pol(rewards, states, next_states, policy_net, policy_optimizer, value_net, log_probs, gam=0.99)

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(policy_net.state_dict(), 'best_policy_model.pth')
        torch.save(value_net.state_dict(), 'best_value_model.pth')
        print(f"New best model saved with reward: {best_reward}")

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
