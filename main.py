import numpy as np
from scipy.spatial.distance import cityblock

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
  def __init__(self, num_blocks, location):
    self.num_blocks = num_blocks
    # location is a tuple of 3 ints
    self.location = location

# Pickup and DropOff are children classes of the Cell parent class
class PickUp(Cell): 
  def __init__(self, num_blocks, location):
    # super() automatically inherit the methods and properties from its parent
    super().__init__(num_blocks, location) 
    # remaining
    def is_valid(num_blocks):
      return num_blocks != 0

class DropOff(Cell): 
  def __init__(self, num_blocks, location):
    super().__init__(num_blocks, location) 
    # capacity 
    def is_valid(num_blocks):
      return num_blocks != 5

# keep track of env -> make an environment class

var_alpha = 0.3
var_lambda = 0.5

# initialized
fem_agent = Agent((1,1,1), (3,2,3), 0, 0) # her pos, his pos, reward, have_block
male_agent = Agent((3,2,3), (1,1,1), 0, 0) 
# cells
pickup1 = PickUp(10, (3,3,2)) # number of blocks currently held, location of the cells
pickup2 = PickUp(10, (2,2,1))
dropoff1 = DropOff(0, (1,1,3))
dropoff2 = DropOff(0,(3,2,3))
dropoff3 = DropOff(0,(1,1,2))
dropoff4 = DropOff(0, (3,1,1))
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

#pick up: +14, drop off: +14, risky: -2, path: -1

Layer_1 = [[-1, -1, -1], [-1, 'P', -1], ['D', 'R', -1]]
Layer_2 = [['D', -1, -1], [-1, 'R', -1], [-1, -1, 'P']]
Layer_3 = [['D', -1, -1], [-1, -1, -1], [-1, 'D', -1]]

#D: drop-off, P: pick-up, R: risky

Layer_1 = np.array(Layer_1).reshape((3,3))
Layer_2 = np.array(Layer_2).reshape((3,3))
Layer_3 = np.array(Layer_3).reshape((3,3))
print(Layer_1)
print(Layer_2)
print(Layer_3)


# Manhattan distance formula: 
  # d = |x1 - x2| + |y1 - y2|
# Luckily, scipy has a library to compute the City Block (Manhattan) distance.
manhat_distance = cityblock(fem_agent.current_pos,male_agent.current_pos)
print('Manhattan Distance between', fem_agent.current_pos, 'and', male_agent.current_pos, 'is', manhat_distance)
