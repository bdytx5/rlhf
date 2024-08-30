import torch
import gym
import torch.nn as nn
import torch
import gym

# class PolicyNet(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, action_dim)
#         )

#     def forward(self, state):
#         return torch.softmax(self.fc(state), dim=1)


# class ValueNet(nn.Module):
#     def __init__(self, state_dim):
#         super(ValueNet, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, state):
#         return self.fc(state)

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
        return torch.softmax(self.fc(state), dim=1)
    
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
    
    
def infer(env, policy_net):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()  # Display the environment
        
        # Ensure state is in the expected format
        if isinstance(state, tuple):
            state = state[0]  # Extract the actual state if it's wrapped in a tuple

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        action = torch.argmax(action_probs, dim=1).item()
            
        # next_state, reward, done, _ = env.step(action)
        next_state, reward, done, _, _ = env.step(action)

        total_reward += reward

        # Update the state
        state = next_state

    return total_reward


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNet(state_dim=state_dim, action_dim=action_dim)
    value_net = ValueNet(state_dim=state_dim)

    # Load the best saved models
    policy_net.load_state_dict(torch.load('best_policy_model.pth'))
    value_net.load_state_dict(torch.load('best_value_model.pth'))

    policy_net.eval()
    value_net.eval()

    # Run inference
    total_reward = infer(env, policy_net)
    print(f"Inference run - Total Reward: {total_reward}")

    env.close()  # Close the environment after rendering
