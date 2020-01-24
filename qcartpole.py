from __future__ import print_function
import os
import neat
import gym
from time import sleep

class Cart(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.observation = self.env.reset()
        self.brain = []

def eval_genomes(genomes, config):
    ge = []
    carts = []
    dones = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        carts.append(Cart())
        carts[-1].brain = net
        genome.fitness = 0
        ge.append(genome)
        dones.append(False)

    score = 0
    h = 0

    while dones:
        for _, cart in enumerate(carts):
            if ge[carts.index(cart)].fitness >= 0 :
                cart.env.render()
            else:
                pass
            inputs = cart.observation.copy()
            output = round(cart.brain.activate((inputs[0], inputs[1], inputs[2], inputs[3]))[0])
            action = output
            observations, reward, done, info = cart.env.step(action)
            ge[carts.index(cart)].fitness += 1
            if done:
                cart.env.close()
                dones.remove(False)
                ge[carts.index(cart)].fitness -= 5
                carts.remove(cart)
    sleep(0.5)

def run(config_file):
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_file)
    # Create the population, wich is the top-level object for NEAT run
    p = neat.Population(config)

    # add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations
    winner = p.run(eval_genomes, 1500)
    # Display the winning genome
    print(f'\nBest genomes: \n{winner}')

    # show output of the most fit genome against training data
    print('\nOutput: ')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


if __name__ == '__main__':
    localdir = os.path.dirname(__file__)
    configpath = os.path.join(localdir, 'config-feedforward.txt')
    run(configpath)
