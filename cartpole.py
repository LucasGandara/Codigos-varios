import gym
env = gym.make('CartPole-v0')
env.reset()
from os import system
from math import pi

#Random actions

for i_episode in range(20):
    observation = env.reset()
    observation[2] *= 180/pi
    for t in range(100):
        env.render()
        print(observation)
        system('pause')
        observation, rew, done, info = env.step(env.action_space.sample()) # random action
        observation[2] *= 180/pi
    if done:
        print(f'Episode finished after {t+1} timesteps')
        break
env.close()

#Spaces
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
