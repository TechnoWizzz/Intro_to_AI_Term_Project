import gym
import minerl
######################################

# Sample some data from the dataset!
data = minerl.data.make("MineRLNavigateDense-v0")

agent_1 = agent()
switch = 0

# Iterate through a single epoch using sequences of at most 32 steps
for obs, rew, done, act in data.seq_iter(num_epochs=1, batch_size=32):
    
    if switch = 0:
        switch = 1
        state = obs
        reward = rew
        done_ = done
        action = act
        pass
    
    else:
        while not done:
            agent_1.update_mem(state, action, reward, obs, done)
            agent_1.train()
            state = obs
            total_reward += reward
            reward = rew
            action = act
            done_ = done
        
    if done:
      print("total reward after {} episode is {} and epsilon is {}".format(s, total_reward, agent_1.epsilon))