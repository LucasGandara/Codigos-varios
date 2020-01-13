import pygame
from pygame.locals import QUIT
import sys
pygame.init()
WIDTH = 500
HEIGHT = 480 
screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
pygame.display.set_caption('pi with collisions')
pygame.font.init()

bounces = pygame.font.SysFont('Arial', 18, False)

counter = 0
clock = pygame.time.Clock()
wall = 30
floor = 400
red = (255, 0, 0)
blue = (0, 0, 255)

class Block(object):
    def __init__(self, size, XY, mass, velocity, color):
        self.x = XY[0]
        self.y = XY[1]
        self.size = size
        self.mass = mass
        self.velocity = velocity
        self.rect = pygame.Rect(self.x, self.y, size, size)
        self.color = color

    def collision(self, otherblock):
        if self.x + self.size < otherblock.x or self.x > otherblock.x + otherblock.size:
            return False
        else:
            return True
            print('collision')

    def collide_wall(self):
        if self.x <= wall:
            self.velocity *= -1
            return True

    def bounce(self, otherblock):
        sumM = self.mass + otherblock.mass
        newV = (self.mass - otherblock.mass) / sumM * self.velocity
        newV += (2 * otherblock.mass/ sumM) * otherblock.velocity
        return newV
    
    def move(self):
        self.x += self.velocity / 100
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def draw(self, win):
        pygame.draw.rect(win, (self.color), self.rect)

def redrawgamewindow():
    screen.fill((0, 0, 0))
    # Draw the lines for the floor and the wall
    pygame.draw.line(screen, (255, 255, 255), (wall, floor), (500, floor)) #Horizontal
    pygame.draw.line(screen, (255, 255, 255), (wall, 0), (wall, floor))      #Vertical
    #print the number of bounces
    screen.blit(bounces.render(f'{counter}', True, (255, 255, 255)), (wall, floor + 20))

    #Draw the rectangles
    big_rect.draw(screen)
    small_rect.draw(screen)

    pygame.display.update()

#Ceate the rectangles
big_rect = Block(60, [wall + 40, floor - 60], 10000000000, -1/10, blue)
small_rect = Block(10, [wall + 20, floor - 10], 1, 0, red)

gaming = True
while gaming: 
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit(0)

    if small_rect.collision(big_rect):
        counter += 1    
        v1 = small_rect.bounce(big_rect)
        v2 = big_rect.bounce(small_rect)
        big_rect.velocity = v2
        small_rect.velocity = v1
    
    if small_rect.collide_wall():
        counter += 1    

    big_rect.move()
    small_rect.move()

    redrawgamewindow()