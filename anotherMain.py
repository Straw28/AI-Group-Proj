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
# Module: Action class --> actually takes the action
world = [[[0, 0, 0], [0, 10, 0],  [5, 0, 0]],       # 1
         [[5, 0, 0], [0, 0, 0],   [0, 0, 10]],       # 2
         [[5, 0, 0], [0, 0, 0],   [0, 5, 0]]]        # 3

world = np.array(world)

# Module: Reward Class
# Maps each state-action pair to a numerical reward signal, which the agent uses to update its policy and improve its decision-making over time.

def printQTable(qtable):
    for a in range(len(q.Qtable)):
        for b in range(len(q.Qtable[a])):
            for c in range(len(q.Qtable[a][b])):
                for d in range(len(q.Qtable[a][b][c])):
                    for e in range(len(q.Qtable[a][b][c][d])):
                        print("Q value at 0: ", q.Qtable[a][b][c][d][e])


def printWorld():

    for i in range(world.shape[0]):
        print("Layer", i+1, ":")
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                print(world[i][j][k], end=" ")
            print()
        print()


class Reward:
    # actions = Action()

    def canPickUp(self, futureAgent, action, currentAgent, world):
        if currentAgent.have_block == False:
            if futureAgent.current_pos == (0,1,1) and world[0, 1, 1] > 0:
                return True
            elif futureAgent.current_pos == (1, 2, 2) and world[1, 2, 2] > 0:
                return True
        return False

    def canDropOff(self, futureAgent, action, currentAgent, world):
        if currentAgent.have_block == True:
            if futureAgent.current_pos == (1, 0, 0) and world[1, 0, 0] > 0:
                return True
            elif futureAgent.current_pos == (2, 0, 0) and world[2, 0, 0] > 0:
                return True
            elif futureAgent.current_pos == (0, 0, 2) and world[0, 0, 2] > 0:
                return True
            elif futureAgent.current_pos == (2, 1, 2) and world[2, 1, 2] > 0:
                return True
        return False

    def isRisky(self, futureAgent, action, currentAgent, world):
        if futureAgent.current_pos == (1,1,1) or futureAgent.current_pos == (0,1,2):
            return True
        return False

    def rewardReturn(self, futureAgent, action, currentAgent, world):
        if self.canPickUp(futureAgent, action, currentAgent, world) or self.canDropOff(futureAgent, action, currentAgent, world):
            return 14
        elif self.isRisky(futureAgent, action, currentAgent, world):
            return -2
        else:
            return -1


class Action:
    rewards = Reward()
    
    def takeDirection(self, agent, agent2, world, direction):
        # print("world before taking action")
        # printWorld()
        agentreward = 0
        oldAgent = agent
        if direction == 0:
            agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2])
            # print("Does the agent have a block? (inside the if): ", agent.have_block)
            # print("this is the agent positions after: ", agent.current_pos)
            # should we check this here? does it check if it has a block or not?
            agentreward = self.rewards.rewardReturn(agent, 0, oldAgent, world)
            if agentreward == 14:  # reward returns 14 if you're able to pickup or drop off successfully
                world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]] -= 1
                if self.rewards.canPickUp(agent, 0, oldAgent, world):
                    agent.have_block = 1
                else:
                    agent.have_block = 0
            agent2.other_pos = agent.current_pos

        elif direction == 1:
            agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2])
            # print("Does the agent have a block? (inside the if): ", agent.have_block)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agentreward = self.rewards.rewardReturn(agent, 0, oldAgent, world)
            if agentreward == 14:  # reward returns 14 if you're able to pickup or drop off successfully
                world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]] -= 1
                if self.rewards.canPickUp(agent, 0, oldAgent, world):
                    agent.have_block = 1
                else:
                    agent.have_block = 0
            agent2.other_pos = agent.current_pos

        elif direction == 2:
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2])
            # print("Does the agent have a block? (inside the if): ", agent.have_block)
            # print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agentreward = self.rewards.rewardReturn(agent, 0, oldAgent, world)
            if agentreward == 14:  # reward returns 14 if you're able to pickup or drop off successfully
                world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]] -= 1
                if self.rewards.canPickUp(agent, 0, oldAgent, world):
                    agent.have_block = 1
                else:
                    agent.have_block = 0
            agent2.other_pos = agent.current_pos

        elif direction == 3:
            agent.current_pos = ( agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2])
            # print("Does the agent have a block? (inside the if): ", agent.have_block)
            # print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agentreward = self.rewards.rewardReturn(agent, 0, oldAgent, world)
            if agentreward == 14:  # reward returns 14 if you're able to pickup or drop off successfully
                world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]] -= 1
                if self.rewards.canPickUp(agent, 0, oldAgent, world):
                    agent.have_block = 1
                else:
                    agent.have_block = 0
            agent2.other_pos = agent.current_pos

        elif direction == 4:
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1)
            # print("Does the agent have a block? (inside the if): ", agent.have_block)
            # print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agentreward = self.rewards.rewardReturn(agent, 0, oldAgent, world)
            if agentreward == 14:  # reward returns 14 if you're able to pickup or drop off successfully
                world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]] -= 1
                if self.rewards.canPickUp(agent, 0, oldAgent, world):
                    agent.have_block = 1
                else:
                    agent.have_block = 0
            agent2.other_pos = agent.current_pos

        elif direction == 5:
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1)
            # print("Does the agent have a block? (inside the if): ", agent.have_block)
            # print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agentreward = self.rewards.rewardReturn(agent, 0, oldAgent, world)
            if agentreward == 14:  # reward returns 14 if you're able to pickup or drop off successfully
                world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]] -= 1
                if self.rewards.canPickUp(agent, 0, oldAgent, world):
                    agent.have_block = 1
                else:
                    agent.have_block = 0
            agent2.other_pos = agent.current_pos

        # print("In Action Class taking direction, direction: ", direction)
        # print("world after: ")
        # printWorld()

        agent.reward += agentreward
# Module: isValid --> checks if a move is valid or not


class isValid:
    # checks for out of bounds & checks for if two agents are in the same block returns an array with valid moves
    # this function tells us what direction the agent is currently able to take
    temp = (0, 0, 0)

    def directionParser(self, agent):
        dirArray = [0, 1, 2, 3, 4, 5]
        # [up, down, north, south, east, west]
        if agent.current_pos[0] - 1 < 0 or (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2]) == agent.other_pos:
            # temp = (agent.current_pos[0] - 1, agent.current_pos[1],agent.current_pos[2]) or
            # if agent.have_block == 0 and temp in dropoffArray:
            dirArray.remove(0)  # up
        if agent.current_pos[0] + 1 > 2 or (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2]) == agent.other_pos:
            dirArray.remove(1)  # down
        if agent.current_pos[1] - 1 < 0 or (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2]) == agent.other_pos:
            dirArray.remove(2)  # north
        if agent.current_pos[1] + 1 > 2 or (agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2]) == agent.other_pos:
            dirArray.remove(3)  # south
        if agent.current_pos[2] + 1 > 2 or (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1) == agent.other_pos:
            dirArray.remove(4)  # east
        if agent.current_pos[2] - 1 < 0 or (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1) == agent.other_pos:
            dirArray.remove(5)  # west

        return dirArray


# don't need a state class bc everything is in agent or cells
class Agent:
    def __init__(self, current_pos, other_pos, reward, have_block):
        self.current_pos = current_pos  # tuple [z,y,x]
        # the other agent's position
        self.other_pos = other_pos
        # cumulative
        self.reward = reward
        # boolean
        self.have_block = have_block  # 0 = no block 1 = block

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


class Qtable:
    Qtable = np.zeros((3, 3, 3, 2, 6))  # always starts off at zero

    # Q[Z][Y][X][block or no block][action]  <-- This is how the q-table is set up

    # formula Q(S_new, a) = Q(S_old, a) + alpha(reward + gamma * max(q-value of all states given the action) - Q(S_old, a))
    # explore first (that's where the 500 steps come in)
    # hence why we call prandom because it randomly decides what future state to go to
    rewards = Reward()
    # Q(s,a) = (1-alpha) * Q(a,s) + alpha*[my_reward.rewardReturn() + gamma * max(Q(a', s'))]

    def qLearning(self, m_agent, f_agent, world, var_gamma, var_alpha, num_steps):
        p = Policy()  # <--- This lets us access the policies
        a = isValid()
        q = Qtable()  # <--- This lets us access the qtable
        # <--- here we iterate through the number of steps per experitment
        for i in range(num_steps):
            old_state_m = m_agent.current_pos
            old_state_f = f_agent.current_pos  # <--- memorize what the past

            # running the policy for the male agent
            m = p.PRandom(m_agent, f_agent, world)
            # running the policy for the female agent
            f = p.PRandom(f_agent, m_agent, world)
            # remember I update the rewards, have block(for agents), num_blocks(for pickup/dropoff) and positions in the policy so no need to do in this function

            new_state_m = m_agent.current_pos
            new_state_f = f_agent.current_pos

            m_future_directions = a.directionParser(m_agent)
            m_future_directions_qvalue = []
            for direction in m_future_directions:
                if m_agent.have_block == 0:
                    m_future_directions_qvalue.append(
                        q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][0][direction])
                elif m_agent.have_block == 1:
                    m_future_directions_qvalue.append(
                        q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][1][direction])

            f_future_directions = a.directionParser(f_agent)
            f_future_directions_qvalue = []
            for direction in f_future_directions:
                if f_agent.have_block == 0:
                    f_future_directions_qvalue.append(
                        q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][0][direction])
                elif f_agent.have_block == 1:
                    f_future_directions_qvalue.append(
                        q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][1][direction])

            if m_agent.have_block == 0:
                q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][0][m] = q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][0][m] + var_alpha * (
                    world[new_state_m[0]][new_state_m[1]][new_state_m[2]] + var_gamma * max(m_future_directions_qvalue) - q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][0][m])
            elif m_agent.have_block == 1:
                q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][1][m] = q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][1][m] + var_alpha * (
                    world[new_state_m[0]][new_state_m[1]][new_state_m[2]] + var_gamma * max(m_future_directions_qvalue) - q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][1][m])

            # go to the beginning of the qtable class
            if f_agent.have_block == 0:
                q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][0][f] = q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][0][f] + var_alpha * (
                    world[new_state_f[0]][new_state_f[1]][new_state_f[2]] + var_gamma * max(f_future_directions_qvalue) - q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][0][f])
            elif f_agent.have_block == 1:
                q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][1][f] = q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][1][f] + var_alpha * (
                    world[new_state_f[0]][new_state_f[1]][new_state_f[2]] + var_gamma * max(f_future_directions_qvalue) - q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][1][f])

# Module: Policy class
class Policy:
    q = Qtable()
    is_it_valid = isValid()
    myaction = Action()
    rewards = Reward()
    # pickups = PickUp()
    # dropoffs = DropOff()
   # rewards = Reward(agent, action, agent, pickups, dropoffs)
    
 
  # PRandom --> it checks if pickup or drop off is possible in the current state. If it's not then
    def PRandom(self, agent, agent2, world):  # 0 0 0
        directions = self.is_it_valid.directionParser(agent)
        for direction in directions:  # this is to check if there is a pick up or drop off available
            futureAgent = agent
            self.myaction.takeDirection(futureAgent, agent2, world, direction)
            # print(f"Here's the agents new position: {futureAgent.current_pos}")
            if self.rewards.rewardReturn(futureAgent, direction, agent, world) > 0:
                print("I found a reward")
                agent = futureAgent
                return
        #print("this is the agent positions before: ", agent.current_pos)
        r = np.random.choice(directions)
        #print("This is the direction we took: ", r)
        self.myaction.takeDirection(agent, agent2, world, r)  # takes direction
        #print("This is the agent's current position: ", agent.current_pos)

    def PGreedy(self, agent, agent2, world):
        directions = self.is_it_valid.directionParser(agent)
        futureAgent = agent
        self.myaction.takeDirection(futureAgent, agent2, world, direction)
        if self.rewards.rewardReturn(futureAgent, direction, agent, world) > 0:
          print("I found a reward")
          return

    # def PExploit(self, agent, agent2, world):
    #   directions = is_it_valid.directionParser(agent)
    #   print("This are the directions: ", directions)
    #   for direction in directions: #this is to check if there is a pick up or drop off available
    #     if self.pickUpAndDropOffCheck(agent, agent2, direction) != -1:
    #       print("I found a reward")
    #       return


#higher q the better?
        finished = True
        for d in dropoffArray:
            if world[d] > 0:
                finished = False
                break
        if finished:
            return i

        # dropoff1 = DropOff(0, (0, 0, 2), 14)
        # dropoff2 = DropOff(0, (2, 1, 2), 14)
        # dropoff3 = DropOff(0, (0, 0, 1), 14)
        # dropoff4 = DropOff(0, (2, 0, 0), 14)

        return num_steps




var_alpha = 0.3
var_lambda = 0.5

# initialized
# her pos, his pos, reward, have_block
fem_agent = Agent((0, 0, 0), (2, 1, 2), 0, 0)
male_agent = Agent((2, 1, 2), (0, 0, 0), 0, 0)
#test_agent = Agent((1, 1, 2), (0, 0, 0), 0, 0)
# cells
# zyx bc we use pickup and drop off array for the world not the agent. agent is stored as xyz
pickup1 = (0, 1, 1)
pickup2 = (1, 2, 2)
pickupArray = [pickup1, pickup2]

dropoff1 = (1, 0, 0)
dropoff2 = (2, 0, 0)
dropoff3 = (0, 0, 2)
dropoff4 = (2, 1, 2)
dropoffArray = [dropoff1, dropoff2, dropoff3, dropoff4]

# Note that he indexes at one
# risky:(2,2,2),(3,2,1)

# our Q-table, initialized to 0 on purpose
# Q-table has states as rows and acions/operators as columns
# so will be bigger than 3*3
# need to create state space first then Q table
# q_table = np.zeros((3, 3), dtype=int, order='C')
# print("Q-Table")
# print(q_table)

# q = Qtable()
# num_steps = 100
# a = q.qLearning(male_agent, fem_agent, world, var_lambda, var_alpha, num_steps)
# print("Q-Table")
# print(a)
print("world before")
printWorld()
pol = Policy()
for i in range(9000):
    pol.PRandom(male_agent, fem_agent, world)

print("world after")
printWorld()

print("Reward Male agent: ", male_agent.reward)
# print("World: ", world[0, 1, 1])  # zyx

# pick up: +14, drop off: +14, risky: -2, path: -1

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
# manhat_distance = cityblock(fem_agent.current_pos, male_agent.current_pos)
# print('Manhattan Distance between', fem_agent.current_pos, 'and', male_agent.current_pos, 'is', manhat_distance)

# numSteps = 10000

# q = Qtable()
# print("Number of steps till finished: ", q.qLearning(male_agent, fem_agent, world, var_lambda, var_alpha, numSteps))
# print("Male agent position: ", male_agent.current_pos," Female agent position: ", fem_agent.current_pos)
print("printing world: ")
printWorld()