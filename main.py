import numpy as np
from scipy.spatial.distance import cityblock

# Environment --> the whole world
# State --> part of the world, (x, y, z, x’, y’, z’, i, i’, a, b, c, d, e, f)
# TODO: operators class needs to check for out-of-bounds
# TODO: implement out of bounds function in Cell class
# TODO: keep track of env -> make an environment class
# TODO: test if the reward for each operator is correct
# TODO: make the Model class/function in Module (Michelle)

# Module: State class
# Have to figure out and reduce the number of states in state space first to work with Q-learning & SARSA algorithms
# Original Number of States from (x, y, z, x’, y’, z’, i, i’, a, b, c, d, e, f) is 182 million states (too huge)
# 27 (for male x,y,z) * 27 (for female x',y',z') * 2 (male w/ or w/o block) * 2 (female w/ or w/o block) *5*5*5*5 *10*10
# The four 5's are the blocks from four dropoff cells (a, b, c, d) and two 10's are blocks from two pickup cells (e, f)
'''
class State:
  def __init__(x, y, z, x’, y’, z’, i, i’, a, b, c, d, e, f):
    #intitalize all the stuffs
    def currentState(): #returns current state
'''

# Module: Policy class
class Policy:
  def directionParser(self, agent): #this function tells us what direction the agent is currently able to take
    dirArray = [0, 1, 2, 3, 4, 5]
    #[up, down, north, south, east, west]

    if agent.current_pos[0] - 1 < 0 or (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2]) == agent.other_pos:
      dirArray.remove(0) #up
    if agent.current_pos[0] + 1 > 2 or (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2]) == agent.other_pos:
      dirArray.remove(1) #down
    if agent.current_pos[1] - 1 < 0 or (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2]) == agent.other_pos:
      dirArray.remove(2) #north
    if agent.current_pos[1] + 1 > 2 or (agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2]) == agent.other_pos:
      dirArray.remove(3) #south
    if agent.current_pos[2] + 1 > 2 or (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1) == agent.other_pos:
      dirArray.remove(4) #east
    if agent.current_pos[2] - 1 < 0 or (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1) == agent.other_pos:
      dirArray.remove(5) #west

    return dirArray

  def pickUpAndDropOffCheck(self, agent, agent2, direction, pickupArray, dropoffArray):
    #this function checks if there is a pick up/drop off on the location that we might go to
    if direction == 0:  # up
      if agent.have_block == 0:
        for p in pickupArray:
          if (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2]) == p.location and p.is_valid():
            agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2])
            agent.reward += 14
            agent.have_block = 1
            p.num_blocks -= 1
            agent2.other_pos = agent.current_pos
            return direction
      else:
        for d in dropoffArray:
          if (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2]) == d.location and d.is_valid():
            agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2])
            agent.reward += 14
            agent.have_block = 0
            d.num_blocks += 1
            agent2.other_pos = agent.current_pos
            return direction

    elif direction == 1:
      if agent.have_block == 0:
        for p in pickupArray:
          if (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2]) == p.location and p.is_valid():
            agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2])
            agent.reward += 14
            agent.have_block = 1
            p.num_blocks -= 1
            agent2.other_pos = agent.current_pos
            return direction
      else:
        for d in dropoffArray:
          if (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2]) == d.location and d.is_valid():
            agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2])
            agent.reward += 14
            agent.have_block = 0
            d.num_blocks += 1
            agent2.other_pos = agent.current_pos
            return direction

    elif direction == 2:
      if agent.have_block == 0:
        for p in pickupArray:
          if (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2]) == p.location and p.is_valid():
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2])
            agent.reward += 14
            agent.have_block = 1
            p.num_blocks -= 1
            agent2.other_pos = agent.current_pos
            return direction
      else:
        for d in dropoffArray:
          if (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2]) == d.location and d.is_valid():
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2])
            agent.reward += 14
            agent.have_block = 0
            d.num_blocks += 1
            agent2.other_pos = agent.current_pos
            return direction

    elif direction == 3:
      if agent.have_block == 0:
        for p in pickupArray:
          if (agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2]) == p.location and p.is_valid():
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2])
            agent.reward += 14
            agent.have_block = 1
            p.num_blocks -= 1
            agent2.other_pos = agent.current_pos
            return direction
      else:
        for d in dropoffArray:
          if (agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2]) == d.location and d.is_valid():
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2])
            agent.reward += 14
            agent.have_block = 0
            d.num_blocks += 1
            agent2.other_pos = agent.current_pos
            return direction

    elif direction == 4:
      if agent.have_block == 0:
        for p in pickupArray:
          if (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1) == p.location and p.is_valid():
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1)
            agent.reward += 14
            agent.have_block = 1
            p.num_blocks -= 1
            agent2.other_pos = agent.current_pos
            return direction
      else:
        for d in dropoffArray:
          if (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1) == d.location and d.is_valid():
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1)
            agent.reward += 14
            agent.have_block = 0
            d.num_blocks += 1
            agent2.other_pos = agent.current_pos
            return direction
    elif direction == 5:
      if agent.have_block == 0:
        for p in pickupArray:
          if (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1) == p.location and p.is_valid():
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1)
            agent.reward += 14
            agent.have_block = 1
            p.num_blocks -= 1
            agent2.other_pos = agent.current_pos
            return direction
      else:
        for d in dropoffArray:
          if (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1) == d.location and d.is_valid():
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1)
            agent.reward += 14
            agent.have_block = 0
            d.num_blocks += 1
            agent2.other_pos = agent.current_pos
            return direction

    return -1

  def takeDirection(self, agent, agent2, world, direction):
    if direction == 0:
      agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2])
      print("this is the agent positions after: ", agent.current_pos)
      agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
      agent2.other_pos = agent.current_pos
    elif direction == 1:
      agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2])
      print("this is the agent positions after: ", agent.current_pos)
      agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
      agent2.other_pos = agent.current_pos
    elif direction == 2:
      agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2])
      print("this is the agent positions after: ", agent.current_pos)
      agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
      agent2.other_pos = agent.current_pos
    elif direction == 3:
      agent.current_pos = (agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2])
      print("this is the agent positions after: ", agent.current_pos)
      agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
      agent2.other_pos = agent.current_pos
    elif direction == 4:
      agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1)
      print("this is the agent positions after: ", agent.current_pos)
      agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
      agent2.other_pos = agent.current_pos
    elif direction == 5:
      agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1)
      print("this is the agent positions after: ", agent.current_pos)
      agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
      agent2.other_pos = agent.current_pos

  def PRandom(self, agent, agent2, world, pickupArray, dropoffArray):
    directions = self.directionParser(agent)
    print("This are the directions: ", directions)
    for direction in directions: #this is to check if there is a pick up or drop off available
      if self.pickUpAndDropOffCheck(agent, agent2, direction, pickupArray, dropoffArray) != -1:
        print("I found a reward")
        return
    print("this is the agent positions before: ", agent.current_pos)
    r = np.random.choice(directions)
    print("This is the direction we took: ", r)
    self.takeDirection(agent, agent2 ,world, r)

  def PGreedy(self, agent, agent2, world, pickupArray, dropoffArray):
    directions = self.directionParser(agent)
    print("This are the directions: ", directions)
    for direction in directions: #this is to check if there is a pick up or drop off available
      if self.pickUpAndDropOffCheck(agent, agent2, direction, pickupArray, dropoffArray) != -1:
        print("I found a reward")
        return

  def PExploit(self, agent, agent2, world, pickupArray, dropoffArray):
    directions = self.directionParser(agent)
    print("This are the directions: ", directions)
    for direction in directions: #this is to check if there is a pick up or drop off available
      if self.pickUpAndDropOffCheck(agent, agent2, direction, pickupArray, dropoffArray) != -1:
        print("I found a reward")
        return

#Module: Reward Class
  class Reward:  #Maps each state-action pair to a numerical reward signal, which the agent uses to update its policy and improve its decision-making over time.
  
    def __init__(self, futureAgent, action, currentAgent, pickUpCell, dropOffCell):
      # future state (stored within agent)
      self.futureAgent = futureAgent
      # action integer value of moving (up, down, east, west, north, south)
      self.action = action
      # current state (stored within agent)
      self.currentAgent = currentAgent
      
      # adding pickup/dropoff cells to keep track of the number of blocks
      self.pickUpCell = pickUpCell
      self.dropOffCell = dropOffCell

    # returns a bool, checks if the agent is in a pickup cell & does not have a block
    def canPickUp(self): 
      if self.currentAgent.have_block == False:
        if self.futureAgent.current_pos == (2, 2, 1) or self.futureAgent.current_pos == (3, 3, 2):
          if self.pickUpCell.is_valid():
            return True
      return False

    # returns a bool, checks if agent is in a drop off cell and has a block
    def canDropOff(self):
      if self.currentAgent.have_block == True: #FIXME also gotta keep track of if the drop offs are full or not
        if self.futureAgent.current_pos == (1, 1, 2) or self.futureAgent.current_pos == (1, 1, 3) or self.futureAgent.current_pos == (3, 1, 1) or self.futureAgent.current_pos == (3, 2, 3):
          if self.dropOffCell.is_valid():
            return True
      return False

    # returns a bool, checks if agent is in a risky cell
    def isRisky(self):
      if self.futureAgent.current_pos == (2, 2, 2) or self.futureAgent.current_pos == (3, 2, 1):
        return True
      return False

    # returns an integer value of the reward
    def rewardReturn(self): 
      if self.canPickUp() or self.canDropOff():
        return 14
      elif self.isRisky():
        return -2
      else:
        return -1

# don't need a state class bc everything is in agent or cells
class Agent:
  def __init__(self, current_pos, other_pos, reward, have_block):
    self.current_pos = current_pos
    # the other agent's position
    self.other_pos = other_pos
    # cumulative
    self.reward = reward
    # boolean
    self.have_block = have_block

# parent class
class Cell:
  def __init__(self, num_blocks, location, reward=-1):
    self.num_blocks = num_blocks
    # location is a tuple of 3 ints
    self.location = location
    # reward is, by default, -1 for regular cells
    self.reward = reward
    # boolean to check if an operator is out of bounds
    # def is_out_bounds(location):
    #   if (location != )


# Pickup and DropOff are children classes of the Cell parent class
class PickUp(Cell):
  def __init__(self, num_blocks, location, reward):
    # super() automatically inherit the methods and properties from its parent
    super().__init__(num_blocks, location)
    # remaining
    # 14 points is a constant
    self.reward = 14
  def is_valid(num_blocks):
    return num_blocks != 0

class DropOff(Cell):
  def __init__(self, num_blocks, location, reward):
    # inherits parent properties
    super().__init__(num_blocks, location)
    self.reward = 14  # 14 points is a constant
    # capacity
  def is_valid(num_blocks):
    return num_blocks != 5

class Risky(Cell):
  def __init__(self, num_blocks, location, is_out, reward):
    # inherits parent properties
    super().__init__(num_blocks, location, is_out)
    # -2 points is a constant
    self.reward = -2


var_alpha = 0.3
var_lambda = 0.5

# initialized
fem_agent = Agent((0, 0, 0), (2, 1, 2), 0, 0) # her pos, his pos, reward, have_block
male_agent = Agent((2, 1, 2), (0, 0, 0), 0, 0)
# cells
pickup1 = PickUp(10, (2, 2, 1), 14)  # number of blocks currently held, location of the cells
pickup2 = PickUp(10, (1, 1, 0), 14)
dropoff1 = DropOff(0, (0, 0, 2), 14)
dropoff2 = DropOff(0, (2, 1, 2), 14)
dropoff3 = DropOff(0, (0, 0, 1), 14)
dropoff4 = DropOff(0, (2, 0, 0), 14)

pickupArray = [pickup1, pickup2]
dropoffArray = [dropoff1, dropoff2, dropoff3, dropoff4]
# Note that he indexes at one 
# risky:(2,2,2),(3,2,1)

# our Q-table, initialized to 0 on purpose
# Q-table has states as rows and acions/operators as columns
# so will be bigger than 3*3
# need to create state space first then Q table
q_table = np.zeros((3,3), dtype=int, order='C')
print("Q-Table")
print(q_table)

# our environment based on the reward
world = [[[-1, -1, -1], [-1, 14, -1], [14, -2, -1]], #1
         [[14, -1, -1], [-1, -2, -1], [-1, -1, 14]], #2
         [[14, -1, -1], [-1, -1, -1], [-1, 14, -1]]] #3

world = np.array(world)

pol = Policy()
pol.PRandom(male_agent, fem_agent, world, pickupArray, dropoffArray)

#pick up: +14, drop off: +14, risky: -2, path: -1

'''Layer_1 = [[-1, -1, -1], [-1, 'P', -1], ['D', 'R', -1]]
Layer_2 = [['D', -1, -1], [-1, 'R', -1], [-1, -1, 'P']]
Layer_3 = [['D', -1, -1], [-1, -1, -1], [-1, 'D', -1]]

#D: drop-off, P: pick-up, R: risky

Layer_1 = np.array(Layer_1).reshape((3,3))
Layer_2 = np.array(Layer_2).reshape((3,3))
Layer_3 = np.array(Layer_3).reshape((3,3))
print(Layer_1)
print(Layer_2)
print(Layer_3)'''


# Manhattan distance formula:
  # d = |x1 - x2| + |y1 - y2|
# Luckily, scipy has a library to compute the City Block (Manhattan) distance.
manhat_distance = cityblock(fem_agent.current_pos,male_agent.current_pos)
print('Manhattan Distance between', fem_agent.current_pos, 'and', male_agent.current_pos, 'is', manhat_distance)

#I am changing the file
