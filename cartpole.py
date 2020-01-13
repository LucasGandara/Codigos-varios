import gym
env = gym.make('CartPole-v0')
env.reset()
#Random actions

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        observation, rew, done, info = env.step(env.action_space.sample()) # random action
    if done:
        print(f'Episode finished after {t+1} timesteps')
        break
env.close()


#Spaces
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)