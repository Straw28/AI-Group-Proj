import numpy as np
from scipy.spatial.distance import cityblock


# XY
pickup1 = (0, 1)
pickup2 = (1, 2)
pickupArray = [pickup1, pickup2]

dropoff1 = (1, 0)
dropoff2 = (2, 0)
dropoff3 = (0, 2)
dropoff4 = (2, 1)
dropoffArray = [dropoff1, dropoff2, dropoff3, dropoff4]

world = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

world = np.array(world)

for i in pickupArray:
    world[i[0]][i[1]] = 10

for i in dropoffArray:
    world[i[0]][i[1]] = 5


def printWorld(agent_m, agent_f):
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            if (agent_m.current_pos == (i, j)):
                print('M +', world[i][j], end=' ')
            elif (agent_f.current_pos == (i, j)):
                print('F +', world[i][j], end=' ')
            else:
                print(world[i][j], end=" ")
        print()


def agentInfo(agent):
    print("---------------------------------")
    print("Agent Postion: ", agent.current_pos)
    print("Other Agent: ", agent.other_pos)
    print("Reward: ", agent.reward)
    print("Have Block? ", agent.have_block)
    print("---------------------------------")
# don't need a state class bc everything is in agent or cells


class Agent:
    def __init__(self, current_pos, other_pos, reward, have_block):
        self.current_pos = current_pos  # tuple [y,x]
        # the other agent's position
        self.other_pos = other_pos
        # cumulative
        self.reward = reward
        # integer
        self.have_block = have_block


class Reward:
    # actions = Action
    def canPickUp(self, future_agent, world):
        if future_agent.have_block == 0:
            if future_agent.current_pos in pickupArray and world[future_agent.current_pos[0]][future_agent.current_pos[1]] > 0:
                return True
        return False

    def canDropOff(self, future_agent, world):
        if future_agent.have_block == 1:
            if future_agent.current_pos in dropoffArray and world[future_agent.current_pos[0]][future_agent.current_pos[1]] > 0:
                # print("in can drop off")
                return True
        return False

    def isRisky(self, future_agent):
        if future_agent.current_pos == (1, 1) or future_agent.current_pos == (0, 2):
            return True
        return False

    def rewardReturn(self, future_agent, world):
        if self.canPickUp(future_agent, world) or self.canDropOff(future_agent, world):
            return 14
        elif self.isRisky(future_agent):
            return -2
        else:
            return -1


class Action:
    # instantiate a Reward object to be used in Action class

    def deduct_cell_value(self, agent, world):
        rewards = Reward()
        if rewards.canPickUp(agent, world):
            agent.have_block = 1
            print("Picked up")
        elif rewards.canDropOff(agent, world):
            agent.have_block = 0
            print("dropped")
        print("we're subtracting from: ", agent.current_pos)
            # only called if can pick up or drop off
        world[agent.current_pos[0]][agent.current_pos[1]] -= 1
        print("world:", world[agent.current_pos[0]][agent.current_pos[1]])

    def takeDirection(self, agent, agent2, world, direction):
        rewards = Reward()
        a = Action()
        agent_reward = 0
        old_agent = Agent(agent.current_pos, agent.other_pos, agent.reward, agent.have_block)

        if direction == 0:
            agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1])
            agent_reward = rewards.rewardReturn(agent, world)

        elif direction == 1:
            agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1])
            agent_reward = rewards.rewardReturn(agent, world)


        elif direction == 2:
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1)
            agent_reward = rewards.rewardReturn(agent, world)


        elif direction == 3:
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] + 1)
            agent_reward = rewards.rewardReturn(agent, world)


        if agent_reward == 14:
            a.deduct_cell_value(agent, world)
        agent.reward += agent_reward
        agent2.other_pos = agent.current_pos
    
class isValid:
    # checks for out of bounds & checks for if two agents are in the same block returns an array with valid moves
    # this function tells us what direction the agent is currently able to take

    def directionParser(self, agent):
        dirArray = [0, 1, 2, 3]  # [up, down, left, right]
  
        # check for obstacles or other agents in adjacent cells
        if agent.current_pos[0] - 1 < 0 or (agent.current_pos[0] - 1, agent.current_pos[1]) == agent.other_pos:
            dirArray.remove(0)  # up
        if agent.current_pos[0] + 1 > 1 or (agent.current_pos[0] + 1, agent.current_pos[1]) == agent.other_pos:
            dirArray.remove(1)  # down
        if agent.current_pos[1] - 1 < 0 or (agent.current_pos[0], agent.current_pos[1] - 1) == agent.other_pos:
            dirArray.remove(2)  # left
        if agent.current_pos[1] + 1 > 1 or (agent.current_pos[0], agent.current_pos[1] + 1) == agent.other_pos:
            dirArray.remove(3)  # right

        return dirArray

class Policy:
    is_it_valid = isValid()
    myaction = Action()
    rewards = Reward()

    def PRandom(self, agent, agent2, world):  # 0 0 0
        directions = self.is_it_valid.directionParser(agent)
        temp_agent = Agent(agent.current_pos, agent.other_pos,
                           agent.reward, agent.have_block)
        temp_pos = agent.current_pos
        temp_reward= agent.reward
        check = agent.reward
        print("temp agent: ")
        agentInfo(temp_agent)
        for direction in directions:  # this is to check if there is a pick up or drop off available
            temp_agent.current_pos = temp_pos
            temp_agent.reward = temp_reward
            self.myaction.takeDirection(temp_agent, agent2, world, direction) #have block
            if temp_agent.reward > check: #
                agent.current_pos = temp_agent.current_pos
                agent.other_pos = temp_agent.other_pos
                agent.reward = temp_agent.reward
                agent.have_block = temp_agent.have_block
                print("reward found, new agent: ")
                agentInfo(agent)
                return

        r = np.random.choice(directions)
        self.myaction.takeDirection(agent, agent2, world, r)  # takes direction
        print("random")
        agentInfo(agent)
        return


def main():

    p = Policy()
    a = isValid()
    rewards = Reward()

    f_agent = Agent((0, 0), (2, 1), 0, 0)
    m_agent = Agent((2, 1), (0, 0), 0, 0)

    for i in range(10):  # <--- here we iterate through the number of steps per experitment

        old_m = Agent(m_agent.current_pos, m_agent.other_pos,
                      m_agent.reward, m_agent.have_block)
        old_f = Agent(f_agent.current_pos, f_agent.other_pos,
                      f_agent.reward, f_agent.have_block)

        old_state_m = m_agent.current_pos
        old_state_f = f_agent.current_pos

        m = p.PRandom(m_agent, f_agent, world)
        # f = p.PRandom(f_agent, m_agent, world)

        # print("Male Agent: ")
        # agentInfo(m_agent)

        # print("Female Agent: ")
        # agentInfo(f_agent)
        # printWorld(m_agent, f_agent)

        for d in pickupArray:
            print("pickup: ", d)
            print("world at pickup: ", world[d[0]][d[1]])
        print()
        # exp 4 check if the total sum of drop offs = 3 (in our case 17?)
        for d in dropoffArray:
            print("drop: ", d)
            print("world at drop: ", world[d])
        print()
        
        finished = True
        for d in dropoffArray: #100 200 002 212
            if world[d[0]][d[1]] > 0:
                finished = False
                break
        
        if finished:
            print("Done at ", i, "number of steps")
            return
        if i == 9:
            print("ran max num steps")

if __name__ == "__main__":
    main()