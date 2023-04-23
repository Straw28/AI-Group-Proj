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

world = [[[0, 0, 0], [0, 0, 0],  [0, 0, 0]],        # 0
         [[0, 0, 0], [0, 0, 0],   [0, 0, 0]],       # 1
         [[0, 0, 0], [0, 0, 0],   [0, 0, 0]]]        # 2

world = np.array(world)

for i in pickupArray:
    world[i[0]][i[1]][i[2]] = 10

for i in dropoffArray:
    world[i[0]][i[1]][i[2]] = 5

def printQTable(q):
    for a, state in enumerate(q.Qtable):
        for b, actions in enumerate(state):
            for c, other_agent_pos in enumerate(actions):
                for d, has_block in enumerate(other_agent_pos):
                    print(f"Q value at ({a}, {b}, {c}, {d}): {has_block}")

def printWorld(agent_m, agent_f):
    for i in range(world.shape[0]):
        print("Layer", i+1, ":")
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if(agent_m.current_pos == (i,j,k)):
                    print('M +', world[i][j][k],end= ' ')
                elif(agent_f.current_pos == (i,j,k)):
                    print('F +', world[i][j][k], end= ' ')
                else:
                    print(world[i][j][k], end=" ")
            print()
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
        self.current_pos = current_pos  # tuple [z,y,x]
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
            if future_agent.current_pos in pickupArray and world[future_agent.current_pos[0]][future_agent.current_pos[1]][future_agent.current_pos[2]]> 0:
                return True
        return False

    def canDropOff(self, future_agent, world):
        if future_agent.have_block == 1:
            if future_agent.current_pos in dropoffArray and world[future_agent.current_pos[0]][future_agent.current_pos[1]][future_agent.current_pos[2]] > 0:
                #print("in can drop off")
                return True
        return False

    def isRisky(self, future_agent):
        if future_agent.current_pos == (1,1,1) or future_agent.current_pos == (0,1,2):
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
    rewards = Reward()

    def deduct_cell_value(self, agent, old_agent, world):
        if self.rewards.canPickUp(agent, world):
            agent.have_block = 1
        elif self.rewards.canDropOff(agent, world):
            agent.have_block = 0
            print("dropped")
        
        world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]] -= 1
        print("world:", world[agent.current_pos[0]][agent.current_pos[1]][agent.current_pos[2]])
    
    def takeDirection(self, agent, agent2, world, direction):
        agent_reward = 0
        old_agent = Agent(agent.current_pos, agent.other_pos, agent.reward, agent.have_block)

        if direction == 0:
            agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1], agent.current_pos[2])
            agent_reward = self.rewards.rewardReturn(agent, world)
            if agent_reward == 14:
               self.deduct_cell_value(agent, old_agent, world) 
            

        elif direction == 1:
            agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1], agent.current_pos[2])
            agent_reward = self.rewards.rewardReturn(agent, world)
            if agent_reward == 14:
              self.deduct_cell_value(agent, old_agent, world) 
            

        elif direction == 2:
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1, agent.current_pos[2])
            agent_reward = self.rewards.rewardReturn(agent, world)
            if agent_reward == 14:
               self.deduct_cell_value(agent, old_agent, world) 
            

        elif direction == 3:
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1] + 1, agent.current_pos[2])
            agent_reward = self.rewards.rewardReturn(agent, world)
            if agent_reward == 14:
               self.deduct_cell_value(agent, old_agent, world) 
            

        elif direction == 4:
            agent.current_pos = (agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] + 1)
            agent_reward = self.rewards.rewardReturn(agent, world)
            if agent_reward == 14:
               self.deduct_cell_value(agent, old_agent, world)
            

        elif direction == 5:
            agent.current_pos = (
                agent.current_pos[0], agent.current_pos[1], agent.current_pos[2] - 1)
            agent_reward = self.rewards.rewardReturn(agent, world)
            if agent_reward == 14:
               self.deduct_cell_value(agent, old_agent, world) 
            
        agent.reward += agent_reward
        agent2.other_pos = agent.current_pos 


# Checks if a move is valid or not
class isValid:
    # checks for out of bounds & checks for if two agents are in the same block returns an array with valid moves
    # this function tells us what direction the agent is currently able to take

    def directionParser(self, agent):
        dirArray = [0, 1, 2, 3, 4, 5]
        for i in agent.current_pos:
            if i < 0 or i > 2:
                print("invalid location")
                return 
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
        q = Qtable()
        rewards = Reward()

        for i in range(num_steps):
            old_m = Agent(m_agent.current_pos, m_agent.other_pos, m_agent.reward, m_agent.have_block)
            old_f = Agent(f_agent.current_pos, f_agent.other_pos, f_agent.reward, f_agent.have_block)
           
            # print("old agent: ")
            # agentInfo(old_m)
            # agentInfo(old_f)

            old_state_m = m_agent.current_pos
            old_state_f = f_agent.current_pos
           
            if i < 500:
                # running the policy for the male agent
                m = p.PRandom(m_agent, f_agent, world)
                # running the policy for the female agent
                f = p.PRandom(f_agent, m_agent, world)
            else:
                m = p.PGreedy(m_agent, f_agent, world)
                f = p.PGreedy(f_agent, m_agent, world)  
            
            agentInfo(m_agent)
            agentInfo(f_agent)
            
            printWorld(m_agent, f_agent)
            #updated positions of male and female agent post policy
            new_state_m = m_agent.current_pos
            new_state_f = f_agent.current_pos
            #male agent 
            m_future_directions = a.directionParser(m_agent)    #valid directions for the new male agent
            m_future_directions_qvalue = []                     #list stores q values for surrounding directions
            
            for direction in m_future_directions:
                if m_agent.have_block == 0:
                    m_future_directions_qvalue.append(q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][0][direction])
                elif m_agent.have_block == 1:
                    m_future_directions_qvalue.append(q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][1][direction])

            #female agent
            f_future_directions = a.directionParser(f_agent)
            f_future_directions_qvalue = []
            
            for direction in f_future_directions:
                if f_agent.have_block == 0:
                    f_future_directions_qvalue.append(q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][0][direction])
                elif f_agent.have_block == 1:
                    f_future_directions_qvalue.append(q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][1][direction])

            #actually plugging into equation
            #Q(s,a) = (1-alpha) * Q(a,s) + alpha*[my_reward.rewardReturn() + gamma * max(Q(a', s'))]
            if m_agent.have_block == 0:                                                                                     
                q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][0][m] +=  q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][0][m] + var_alpha * (                             
                    rewards.rewardReturn(m_agent, world)+ var_gamma * max(m_future_directions_qvalue) - q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][0][m])
            elif m_agent.have_block == 1:
                q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][1][m] +=  q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][1][m] + var_alpha * (                             
                    rewards.rewardReturn(m_agent, world)+ var_gamma * max(m_future_directions_qvalue) - q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][1][m])

            # go to the beginning of the qtable class
            if f_agent.have_block == 0:                                                                                     
                q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][0][f] +=  q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][0][f] + var_alpha * (                             
                    rewards.rewardReturn(f_agent, world)+ var_gamma * max(f_future_directions_qvalue) - q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][0][f])
            elif f_agent.have_block == 1:
                q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][1][f] +=  q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][1][f] + var_alpha * (                             
                    rewards.rewardReturn(f_agent, world)+ var_gamma * max(f_future_directions_qvalue) - q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][1][f])
            print()
            for d in pickupArray:
                print("pickup: ", d)
                print("world at pickup: ", world[d[0]][d[1]][d[2]])
            print()
            #exp 4 check if the total sum of drop offs = 3 (in our case 17?)
            for d in dropoffArray:
                print("drop: ", d)
                print("world at drop: ", world[d])
            print()
            finished = True
            for d in dropoffArray:
                if world[d[0]][d[1]] [d[2]]> 0:
                    finished = False
                    break
            if finished:
                print ("Done!")
                return i
            
            if i == num_steps - 1:
                return i



    def SARSA(self, m_agent, f_agent, world, var_gamma, var_alpha, num_steps):
        p = Policy()  
        a = isValid()
        rewards = Reward()
        
        for i in range(num_steps):  # <--- here we iterate through the number of steps per experitment
   
            old_m = Agent(m_agent.current_pos, m_agent.other_pos, m_agent.reward, m_agent.have_block)
            old_f = Agent(f_agent.current_pos, f_agent.other_pos, f_agent.reward, f_agent.have_block)

            old_state_m = m_agent.current_pos
            old_state_f = f_agent.current_pos

            if i < 500:
                # running the policy for the male agent
                m = p.PRandom(m_agent, f_agent, world)
                # running the policy for the female agent
                f = p.PRandom(f_agent, m_agent, world)
            else:
                m = p.PGreedy(m_agent, f_agent, world)
                f = p.PGreedy(f_agent, m_agent, world)  
            
            printWorld(m_agent, f_agent)
            
            new_state_m = m_agent.current_pos
            new_state_f = f_agent.current_pos

            m_future_directions = a.directionParser(m_agent) 
            m_future_directions_qvalue = []

            for direction in m_future_directions:
                if m_agent.have_block == 0:
                    m_future_directions_qvalue.append(self.Qtable[new_state_m][0][direction])
                elif m_agent.have_block == 1:
                    m_future_directions_qvalue.append(self.Qtable[new_state_m][1][direction])

            f_future_directions = a.directionParser(f_agent)
            f_future_directions_qvalue = []

            for direction in f_future_directions:
                if f_agent.have_block == 0:
                    f_future_directions_qvalue.append( self.Qtable[new_state_f][0][direction])
                elif f_agent.have_block == 1:
                    f_future_directions_qvalue.append(self.Qtable[new_state_f][1][direction])

            if m_agent.have_block == 0: #sending the old agent twice to the return rewards func bc we just want the reward at that single state
                self.Qtable[new_state_m][0][m] += self.Qtable[old_state_m][0][m] + var_alpha*(rewards.rewardReturn(old_m, world) + var_gamma*self.Qtable[new_state_m][0][m] - self.Qtable[old_state_m][0][m])
            elif m_agent.have_block == 1:
                self.Qtable[new_state_m][1][m] += self.Qtable[old_state_m][1][m] + var_alpha*(rewards.rewardReturn(old_m, world)  + var_gamma*self.Qtable[new_state_m][1][m] - self.Qtable[old_state_m][1][m])

            # go to the beginning of the qtable class
            if f_agent.have_block == 0:
                self.Qtable[new_state_f][0][f] = self.Qtable[old_state_f][0][f] + var_alpha*(rewards.rewardReturn(old_f, world)+ var_gamma*self.Qtable[new_state_f][0][f] - self.Qtable[old_state_f][0][f])
            elif f_agent.have_block == 1:
                self.Qtable[new_state_f][1][f] = self.Qtable[old_state_f][1][f] + var_alpha*(rewards.rewardReturn(old_f, world) + var_gamma*self.Qtable[new_state_f][1][f] - self.Qtable[old_state_f][1][f])

            finished = True
            for d in dropoffArray:
                if world[d] > 0:
                    finished = False
                    break
            if finished:
                print ("Done!")
                return i
            
            if i == num_steps - 1:
                return i
   


class Policy:  
    q = Qtable()
    is_it_valid = isValid()
    myaction = Action()
    rewards = Reward()


  # Checks if pick up or drop off is possible in the current state.

    def PRandom(self, agent, agent2, world):  # 0 0 0
        directions = self.is_it_valid.directionParser(agent)
        temp_agent = Agent(agent.current_pos, agent.other_pos, agent.reward, agent.have_block)
        temp_pos = agent.current_pos
        temp_reward= agent.reward
        check = agent.reward
        
        for direction in directions:  # this is to check if there is a pick up or drop off available
            temp_agent.current_pos = temp_pos
            temp_agent.reward = temp_reward
            self.myaction.takeDirection(temp_agent, agent2, world, direction)
            if temp_agent.reward > check:    
                agent.current_pos = temp_agent.current_pos
                agent.other_pos = temp_agent.other_pos
                agent.reward = temp_agent.reward
                agent.have_block = temp_agent.have_block
                return

        r = np.random.choice(directions)
        self.myaction.takeDirection(agent, agent2, world, r)  # takes direction
        return

    def PGreedy(self, agent, agent2, world):
        directions = self.is_it_valid.directionParser(agent)
        temp_agent = Agent(agent.current_pos, agent.other_pos, agent.reward, agent.have_block)
        temp_pos = agent.current_pos
        temp_reward= agent.reward
        check = agent.reward
        for direction in directions:  # this is to check if there is a pick up or drop off available
            temp_agent.current_pos = temp_pos
            temp_agent.reward = temp_reward
            self.myaction.takeDirection(temp_agent, agent2, world, direction)
            if temp_agent.reward > check:    
                agent.current_pos = temp_agent.current_pos
                agent.other_pos = temp_agent.other_pos
                agent.reward = temp_agent.reward
                agent.have_block = temp_agent.have_block
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
        #print("Greedy, Agent end: ", agent.current_pos)
    
    def PExploit(self, agent, agent2, world):
        directions = self.is_it_valid.directionParser(agent)
        future_agent = agent
        for direction in directions:  # this is to check if there is a pick up or drop off available
            if self.rewards.rewardReturn(future_agent, world) > 0:
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

    # temp_agent = Agent((2, 1, 1), (2, 1, 0), 0, 1)
    # remp_agent = Agent((2, 1, 2), (0, 0, 0), 0, 0)

    # a = Action()
    # a.takeDirection(temp_agent, remp_agent, world, 4)
    # agentInfo(temp_agent)
    
    var_alpha = 0.3
    var_lambda = 0.5


    q = Qtable()
    num_steps = 700
    printWorld(male_agent,fem_agent)
    a = q.qLearning(male_agent, fem_agent, world, var_lambda, var_alpha, num_steps)
    
    
    # # print("Q-Table")
    # # printQTable(q)

   
    print("Male agent reward: ", male_agent.reward, " Female agent reward: ", fem_agent.reward)
    # #"number of steps", a)
    print("number steps ", a)
   

if __name__ == "__main__":
    main()