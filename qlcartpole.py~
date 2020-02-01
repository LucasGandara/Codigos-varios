import gym 
import numpy as np
from tqdm import tqdm
import time

#La aplicacion del algoritmo se realizara sobre el entorno Cartpole de gym
env = gym.make('CartPole-v0')
env.reset()

#AI Variables
LR = 0.1
DISCOUNT = 0.95 # How important future actions are
EPISODES = 100000
SHOW_EVERY = 10000

#primer95o se discretiza
# 20 = numero de intervalos
envlow = env.observation_space.low
envlow[1] = -3; envlow[3] = -3
envhigh = env.observation_space.high
envhigh[1] = 3; envhigh[3] = 3
n_intervalos = [51] * len(env.observation_space.low) 
tamaño_intervalos = (envhigh - envlow) / n_intervalos
tamaño_intervalos[1] = 6 / 50
tamaño_intervalos[3] = 6 / 50

# A little bite of randomness
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilo_decay_value = epsilon /  (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#Creamos la tabla de 3 x 3 x 2
q_table = np.random.uniform(low=0, high=1.0, size=(n_intervalos + [env.action_space.n]))

def get_discrete_state(state):
    """ Here we get a number from 0 to 20 for given state:
        the discrete state;
        Discretize the given state """
    state[1] = 3 if state[1] >= 3 else state[1]
    state[1] = -3 if state[1] <= -3 else state[1]
    state[3] = 3 if state[3] >= 3 else state[3]
    state[3] = -3 if state[3] <= -3 else state[3]

    discrete_state = (state - envlow) / tamano_intervalos
    return tuple(discrete_state.astype(np.int))

pbar = tqdm(total = EPISODES, desc='Learning')
for i in range(EPISODES + 1):
    score = 0
    pbar.update(1)
    if i % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    # get the first state of the enviroment
    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        score += reward
        #discretize the new_state
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
            time.sleep(1/10)
        
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            #Apply the q formula
            new_q = (1 - LR) * current_q + LR * (reward + DISCOUNT * max_future_q)
            # Replace the old with the new q value
            q_table[discrete_state + (action, )] = new_q

        discrete_state = new_discrete_state 
    print('Lo logramos!') if score >= 195 else ('no')
    env.close()
