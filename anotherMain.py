import numpy as np
from scipy.spatial.distance import cityblock


#ZYX
pickup1 = (0, 1, 1)
pickup2 = (1, 2, 2)
pickupArray = [pickup1, pickup2]

dropoff1 = (1, 0, 0)
dropoff2 = (2, 0, 0)
dropoff3 = (0, 0, 2)
dropoff4 = (2, 1, 2)
dropoffArray = [dropoff1, dropoff2, dropoff3, dropoff4]

world = [[[0, 0, 0], [0, 10, 0],  [5, 0, 0]],       # 1
         [[5, 0, 0], [0, 0, 0],   [0, 0, 10]],       # 2
         [[5, 0, 0], [0, 0, 0],   [0, 5, 0]]]        # 3

world = np.array(world)


def printQTable(q):
    for a in range(len(q.Qtable)):
        for b in range(len(q.Qtable[a])):
            for c in range(len(q.Qtable[a][b])):
                for d in range(len(q.Qtable[a][b][c])):
                    print(f"Q value at", a, b, c, d,
                          ": ", q.Qtable[a][b][c][d])

def printWorld():

    for i in range(world.shape[0]):
        print("Layer", i+1, ":")
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                print(world[i][j][k], end=" ")
            print()
        print()


# don't need a state class bc everything is in agent or cells
class Agent:
    def __init__(self, current_pos, other_pos, reward, have_block):
        self.current_pos = current_pos  # tuple [z,y,x]
        # the other agent's position
        self.other_pos = other_pos
        # cumulative
        self.reward = reward
        # integer
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

class Reward:
    # actions = Action()

    def canPickUp(self, future_agent, currentAgent, world):
        if currentAgent.have_block == 0:
            if future_agent.current_pos == (0, 1, 1) and world[0, 1, 1] > 0:
                return True
            elif future_agent.current_pos == (1, 2, 2) and world[1, 2, 2] > 0:
                return True
        return False

    def canDropOff(self, future_agent, currentAgent, world):
        if currentAgent.have_block == 1:
            if future_agent.current_pos == (1, 0, 0) and world[1, 0, 0] > 0:
                return True
            elif future_agent.current_pos == (2, 0, 0) and world[2, 0, 0] > 0:
                return True
            elif future_agent.current_pos == (0, 0, 2) and world[0, 0, 2] > 0:
                return True
            elif future_agent.current_pos == (2, 1, 2) and world[2, 1, 2] > 0:
                return True
        return False

    def isRisky(self, future_agent):
        if future_agent.current_pos == (2, 2, 2) or future_agent.current_pos == (3, 2, 1):
            return True
        return False

    def rewardReturn(self, future_agent, currentAgent, world):
        if self.canPickUp(future_agent,  currentAgent, world) or self.canDropOff(future_agent, currentAgent, world):
            return 14
        elif self.isRisky(future_agent):
            return -2
        else:
            return -1


class Action:
    # instantiate a Reward object to be used in Action class
    rewards = Reward()

    def takeDirection(self, agent, agent2, world, direction):
        agent_reward = 0
        old_agent = agent
        if direction == 0:
            agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2])
            #print("this is the agent positions after: ", agent.current_pos)
            # should we check this here? does it check if it has a block or not?
            agent_reward = self.rewards.rewardReturn(agent, old_agent, world)
            # reward returns 14 if you're able to pickup or drop off successfully
            if agent_reward == 14:
                if self.rewards.canPickUp(agent, old_agent, world):
                    agent.have_block = 1
                else:
                    agent_have_block = 0
                world[agent.current_pos] -= 1
                #printWorld()
            agent2.other_pos = agent.current_pos

        elif direction == 1:
            agent.current_pos = (
                agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2])
            #print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agent_reward = self.rewards.rewardReturn(agent, old_agent, world)
            # reward returns 14 if you're able to pickup or drop off successfully
            if agent_reward == 14:
                if self.rewards.canPickUp(agent, old_agent, world):
                    agent.have_block = 1
                else:
                    agent_have_block = 0
                world[agent.current_pos] -= 1
            agent2.other_pos = agent.current_pos

        elif direction == 2:
            agent.current_pos = (
                agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2])
            # print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agent_reward = self.rewards.rewardReturn(agent, old_agent, world)
            if agent_reward == 14:
                if self.rewards.canPickUp(agent, old_agent, world):
                    agent.have_block = 1
                else:
                    agent_have_block = 0
                world[agent.current_pos] -= 1
            agent2.other_pos = agent.current_pos

        elif direction == 3:
            agent.current_pos = (
                agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2])
            # print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agent_reward = self.rewards.rewardReturn(agent, old_agent, world)
            if agent_reward == 14:
                if self.rewards.canPickUp(agent, old_agent, world):
                    agent.have_block = 1
                else:
                    agent_have_block = 0
                world[agent.current_pos] -= 1
            agent2.other_pos = agent.current_pos

        elif direction == 4:
            agent.current_pos = (
                agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1)
            # print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agent_reward = self.rewards.rewardReturn(agent, old_agent, world)
            if agent_reward == 14:
                if self.rewards.canPickUp(agent, old_agent, world):
                    agent.have_block = 1
                else:
                    agent_have_block = 0
                world[agent.current_pos] -= 1
            agent2.other_pos = agent.current_pos

        elif direction == 5:
            agent.current_pos = (
                agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1)
            # print("this is the agent positions after: ", agent.current_pos)
            # agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]]
            agent_reward = self.rewards.rewardReturn(agent, old_agent, world)
            if agent_reward == 14:
                if self.rewards.canPickUp(agent, old_agent, world):
                    agent.have_block = 1
                else:
                    agent_have_block = 0
                world[agent.current_pos] -= 1
            agent2.other_pos = agent.current_pos
        agent.reward += agent_reward 
# Checks if a move is valid or not
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


class Qtable:
    # Q[Z][Y][X][block or no block][action]  <-- This is how the q-table is set up
    # Q(s,a) = (1-alpha) * Q(a,s) + alpha*[my_reward.rewardReturn() + gamma * max(Q(a', s'))]
    Qtable = np.zeros((3, 3, 3, 2, 6))  # always starts off at zero

    def qLearning(self, m_agent, f_agent, world, var_gamma, var_alpha, num_steps):
        p = Policy()  
        a = isValid()
        rewards = Reward()

        for i in range(num_steps):
            old_state_m = m_agent.current_pos
            old_state_f = f_agent.current_pos  # <--- memorize what the past
            old_m = m_agent
            old_f = f_agent
            if i < 500:
                # running the policy for the male agent
                m = p.PRandom(m_agent, f_agent, world)
                # running the policy for the female agent
                f = p.PRandom(f_agent, m_agent, world)
            else:
                m = p.PExploit(m_agent, f_agent, world)
                f = p.PExploit(f_agent, m_agent, world)

            #updated positions of male and female agent post policy
            new_state_m = m_agent.current_pos
            new_state_f = f_agent.current_pos

            #male agent 
            m_future_directions = a.directionParser(m_agent)    #valid directions for the new male agent
            m_future_directions_qvalue = []                     #list stores q values for surrounding directions
            
            for direction in m_future_directions:
                if m_agent.have_block == 0:                     #does not have block
                    m_future_directions_qvalue.append(self.Qtable[new_state_m][0][direction]) #state/ input to the q table
                elif m_agent.have_block == 1:
                    m_future_directions_qvalue.append(self.Qtable[new_state_m][1][direction])

            #female agent
            f_future_directions = a.directionParser(f_agent)
            f_future_directions_qvalue = []
            
            for direction in f_future_directions:
                if f_agent.have_block == 0:
                    f_future_directions_qvalue.append(
                        self.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][0][direction])
                elif f_agent.have_block == 1:
                    f_future_directions_qvalue.append(
                        self.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][1][direction])

            #actually plugging into equation
            #Q(s,a) = (1-alpha) * Q(a,s) + alpha*[my_reward.rewardReturn() + gamma * max(Q(a', s'))]
            if m_agent.have_block == 0:
                self.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][0][m] = self.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][0][m] + var_alpha * (
                    rewards.rewardReturn(m_agent, old_m, world)+ var_gamma * max(m_future_directions_qvalue) - self.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][0][m])
            elif m_agent.have_block == 1:
                self.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][1][m] = self.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][1][m] + var_alpha * (
                    rewards.rewardReturn(m_agent, old_m, world) + var_gamma * max(m_future_directions_qvalue) - self.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][1][m])

            # go to the beginning of the qtable class
            if f_agent.have_block == 0:
                self.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][0][f] = self.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][0][f] + var_alpha * (
                    rewards.rewardReturn(f_agent, old_f, world)+ var_gamma * max(f_future_directions_qvalue) - self.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][0][f])
            elif f_agent.have_block == 1:
                self.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][1][f] = self.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][1][f] + var_alpha * (
                    rewards.rewardReturn(f_agent, old_f, world) + var_gamma * max(f_future_directions_qvalue) - self.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][1][f])

    # the higher the q-value the better
        finished = True
        for d in dropoffArray:
            if world[d] > 0:
                finished = False
                break
            if finished:
                return i
        return num_steps
# Module: Policy class


class Policy:
    q = Qtable()
    is_it_valid = isValid()
    myaction = Action()
    rewards = Reward()
    # pickups = PickUp()
    # dropoffs = DropOff()

  # Checks if pick up or drop off is possible in the current state.

    def PRandom(self, agent, agent2, world):  # 0 0 0
        directions = self.is_it_valid.directionParser(agent)
        for direction in directions:  # this is to check if there is a pick up or drop off available
            future_agent = agent
            self.myaction.takeDirection(future_agent, agent2, world, direction)
            # print(f"Here's the agents new position: {future_agent.current_pos}")
            if self.rewards.rewardReturn(future_agent, agent, world) > 0:
                print("I found a reward")
                agent = future_agent
                return
        # print("this is the agent positions before: ", agent.current_pos)
        r = np.random.choice(directions)
        # print("This is the direction we took: ", r)
        self.myaction.takeDirection(agent, agent2, world, r)  # takes direction
        # print("This is the agent's current position: ", agent.current_pos)

    def PGreedy(self, agent, agent2, world):
        directions = self.is_it_valid.directionParser(agent)
        for direction in directions:  # this is to check if there is a pick up or drop off available
            future_agent = agent
            if self.rewards.rewardReturn(future_agent, agent, world) > 0:
                agent = future_agent
                return

        directionsQvalues = dict()
        state = agent.current_pos
        for direction in directions:
            if agent.have_block == 0:
                directionsQvalues[direction] = self.q.Qtable[state[0]][state[1]][state[2]][0][direction]
            elif agent.have_block == 1:
                directionsQvalues[direction] = self.q.Qtable[state[0]][state[1]][state[2]][1][direction]


        maxdirection = max(directionsQvalues, key = directionsQvalues.get)
        self.myaction.takeDirection(agent, agent2, world, maxdirection)
    
    def PExploit(self, agent, agent2, world):
        directions = self.is_it_valid.directionParser(agent)
        for direction in directions:  # this is to check if there is a pick up or drop off available
            future_agent = agent
            if self.rewards.rewardReturn(future_agent, agent, world) > 0:
                agent = future_agent
                return
        r = np.random.rand()
        if r <= .80:
            self.PGreedy(agent, agent2, world)
        else:
            self.PRandom(agent, agent2, world)





def main():

 
    fem_agent = Agent((0, 0, 0), (2, 1, 2), 0, 0)
    male_agent = Agent((2, 1, 2), (0, 0, 0), 0, 0)

    
    var_alpha = 0.3
    var_lambda = 0.5
    
    # old = Agent((1, 1, 1), (2, 1, 2), 0, 0)
    # # newa = Agent((0, 1, 1), (0, 0, 0), 0, 0)

    # a = Action()
    # a.takeDirection(old, fem_agent, world, 0)
    # r = Reward()
    # print("reward: ", r.rewardReturn(newa, old, world))

    printWorld()

    q = Qtable()
    num_steps = 10000
    a = q.qLearning(male_agent, fem_agent, world, var_lambda, var_alpha, num_steps)
    
    print("Q-Table")
    printQTable(q)

    #printWorld()
    manhat_distance = cityblock(fem_agent.current_pos, male_agent.current_pos)
    print('Manhattan Distance between', fem_agent.current_pos, 'and', male_agent.current_pos, 'is', manhat_distance)


# 
    print("Male agent reward: ", male_agent.reward, " Female agent reward: ", fem_agent.reward)

    printWorld()

if __name__ == "__main__":
    main()