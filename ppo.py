### implementation of a2c 

# valnet, policynet


import torch
import torch.nn as nn
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
#### advantage is produced by the critic (eg the value net) -- A = r + v(s+1) - v(s)

####### critic is trained with td error


# final loss = grad(log policy(a|s))  * advantage)



# def calc_entropy_bonus(oldpol, newpol, s):
#     oldp = oldpol(s)
#     newp = newpol(s)
#     return 0.1*-(oldp*torch.log(oldp/newp)).sum()

def calc_entropy_bonus(newpol, ss):
    # formula is sum over a of (p(a|s)*log(p(a|s)))
    pas = newpol(ss)
    logpas = torch.log(pas)
    return -0.01*(pas * logpas).mean()
    



def calc_l_clip(oldpol, newpol, s, a, eps, A):
    
    pinewa = newpol(s)
    oldpia = oldpol(s)
    pinewa, oldpia = pinewa[range(pinewa.shape[0]), a], oldpia[range(oldpia.shape[0]), a]
    r_theta = pinewa/oldpia
    clp = torch.clamp(r_theta, 1 - eps, 1 + eps)
    right =  clp * A     
    left = r_theta * A
    return torch.min(left, right)





def update_pol(rs, ss,nss, pol, pol_opt, val_net, a_s, gam, oldpol): 
    # loss  log policy(a|s) * (r - val(s))
    # eg loss = (log policy(a|s))  * advantage # policy eg prob of action * quality of action --- so say high prob of 
    # how do we compute the loss here? 
    # our adavantage is essentially our label ? 
    _, ss, nss = torch.tensor(rs), torch.tensor(ss), torch.tensor(nss)

    discounted_rewards = []
    running_add = 0
    for r in rs[::-1]:
        running_add = r + gam * running_add
        discounted_rewards.insert(0, running_add)
    
    rs = torch.tensor(discounted_rewards)
    # A = r + v(s+1) - v(s)
    A = rs + ((gam*val_net(nss)) - val_net(ss) ) # [25] + 
    A = (A - A.mean()) / (A.std() + 1e-8) # NORM????? 

    # logits = pol(ss)    
    # log_action_taken = -torch.log(logits[range(len(a_s)), a_s])
    # ls = log_action_taken * A
    ls = calc_l_clip(oldpol, pol, ss, a_s, eps=0.2, A=A) + calc_entropy_bonus(pol, ss)


    # ls =  * A # [25,4] * [25,25] # seems to be wrong ?? ?
    pol_opt.zero_grad()
    ls.mean().backward()
    pol_opt.step()
    # print(ls.mean())





# updates happen every batch ? 

def update_val(rs, ss, ns, crit, cit_opt, gam=0.99): 
    # loss is r + val(s+1) - val(s)
    _, ss, ns = torch.tensor(rs), torch.tensor(ss), torch.tensor(ns)
    discounted_rewards = []
    running_add = 0
    for r in rs[::-1]:
        running_add = r + gam * running_add
        discounted_rewards.insert(0, running_add)
    
    rs = torch.tensor(discounted_rewards)    
    criterion = nn.MSELoss()
    vals = crit(ss)
    vsplus1s = crit(ns)
    rss = rs + gam*vsplus1s.squeeze()
    vals = vals.squeeze()
    loss = criterion(rss, vals)

    cit_opt.zero_grad()
    loss.backward()
    cit_opt.step()

    





import torch
import copy 
epidsodes = 10000000

env = gym.make('CartPole-v1')
pol, critc = PolicyNet(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n), ValueNet(state_dim=env.observation_space.shape[0])

old_pol = copy.deepcopy(pol)
# make optimizers 
pol_opt, crit_opt = torch.optim.Adam(pol.parameters(), lr=0.001), torch.optim.Adam(critc.parameters(), lr=0.001)
ss, ns, rs = [], [], []
a_s = []
best_reward = -float('inf')

for e in range(epidsodes):
    state = env.reset()
    done = False

    while not done:
        st = state[0] if len(state)==2 else state 
        action_probs = pol(torch.tensor(st, dtype=torch.float32).unsqueeze(0)) # sample from this policy
        action = torch.multinomial(action_probs, num_samples=1).item()

        a_s.append(action)
        next_state, reward, done, _, _ = env.step(action)


        ss.append(st), ns.append(next_state), rs.append(reward)
        state = next_state


    total_reward = sum(rs)

    if e % 128 == 0:
        print(f"Episode {e}, Total Reward: {total_reward}")
        old_pol = copy.deepcopy(pol)
        for e in range(10): 
            update_pol(rs, ss, ns, pol, pol_opt, critc, a_s, gam=0.9, oldpol=old_pol)
        update_val(rs, ss, ns, critc, crit_opt, gam=0.9)
        old_pol = copy.deepcopy(pol)
        
        ss, ns, rs = [], [], []
        a_s = []

    # Save the best models
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(pol.state_dict(), 'best_policy_model.pth')
        torch.save(critc.state_dict(), 'best_value_model.pth')
        print(f"New best model saved with reward: {best_reward}")
