import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make('MountainCar-v0')
env.reset()

#AI variables
LEARNING_RATE = 0.1
DISCOUNT = 0.95 # How important are future actions
EPISODES = 25000
SHOW_EVERY = 2000

# Discretizar el intervalo continuo en 20
discrete_os_size = [20] * len(env.observation_space.low)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

#how random our model is going to be
epsilon = 0.5 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#Create the Q-table
#for every combination of states 20x20 we have 3 possible actions we can take
q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep' : [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    """ Here we get a number from 0 to 20 for given state:
        the discrete state;
        Discretize the given state """
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

pbar = tqdm(total=EPISODES, desc='Entrenando')

for episodes in range(EPISODES): 
    pbar.update(1)
    episodes_reward = 0
    if episodes % SHOW_EVERY == 0:
        print(f'Episode: {episodes}')
        render = True
    else:
        render = False

    # Get the first discrete state by discretize the first one
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon: 
            action = np.argmax(q_table[discrete_state]) # The bes q value in actual state
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episodes_reward += reward 

        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            # Apply the formula
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE *(reward + DISCOUNT * max_future_q)
            # update the q table
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episodes >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episodes_reward)
 
    if not episodes % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episodes)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        #show info
        print(f'Episode: {episodes} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}')

    env.close()
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
