import numpy as np


class Policy:

	def directionParser(
	 self, agent
	):  # this function tells us what direction the agent is currently able to take
		dirArray = [0, 1, 2, 3, 4, 5]
		# [up, down, north, south, east, west]

		if agent.current_pos[0] - 1 < 0 or (agent.current_pos[0] - 1,
		                                    agent.current_pos[1],
		                                    agent.current_pos[2]) == agent.other_pos:
			dirArray.remove(0)  # up
		if agent.current_pos[0] + 1 > 2 or (agent.current_pos[0] + 1,
		                                    agent.current_pos[1],
		                                    agent.current_pos[2]) == agent.other_pos:
			dirArray.remove(1)  # down
		if agent.current_pos[1] - 1 < 0 or (agent.current_pos[0],
		                                    agent.current_pos[1] - 1,
		                                    agent.current_pos[2]) == agent.other_pos:
			dirArray.remove(2)  # north
		if agent.current_pos[1] + 1 > 2 or (agent.current_pos[0],
		                                    agent.current_pos[1] + 1,
		                                    agent.current_pos[2]) == agent.other_pos:
			dirArray.remove(3)  # south
		if agent.current_pos[2] + 1 > 2 or (
		  agent.current_pos[0], agent.current_pos[1],
		  agent.current_pos[2] + 1) == agent.other_pos:
			dirArray.remove(4)  # east
		if agent.current_pos[2] - 1 < 0 or (
		  agent.current_pos[0], agent.current_pos[1],
		  agent.current_pos[2] - 1) == agent.other_pos:
			dirArray.remove(5)  # west

		return dirArray

	def pickUpAndDropOffCheck(self, agent, agent2, direction, pickupArray,
	                          dropoffArray):
		# this function checks if there is a pick up/drop off on the location that we might go to
		if direction == 0:  # up
			if agent.have_block == 0:
				for p in pickupArray:
					if (agent.current_pos[0] - 1, agent.current_pos[1],
					    agent.current_pos[2]) == p.location and p.is_valid(p.num_blocks):
						agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1],
						                     agent.current_pos[2])
						agent.reward += 14
						agent.have_block = 1
						p.num_blocks -= 1
						agent2.other_pos = agent.current_pos
						return direction
			else:
				for d in dropoffArray:
					if (agent.current_pos[0] - 1, agent.current_pos[1],
					    agent.current_pos[2]) == d.location and d.is_valid(num_blocks):
						agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1],
						                     agent.current_pos[2])
						agent.reward += 14
						agent.have_block = 0
						d.num_blocks += 1
						agent2.other_pos = agent.current_pos
						return direction

		elif direction == 1:
			if agent.have_block == 0:
				for p in pickupArray:
					if (agent.current_pos[0] + 1, agent.current_pos[1],
					    agent.current_pos[2]) == p.location and p.is_valid(p.num_blocks):
						agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1],
						                     agent.current_pos[2])
						agent.reward += 14
						agent.have_block = 1
						p.num_blocks -= 1
						agent2.other_pos = agent.current_pos
						return direction
			else:
				for d in dropoffArray:
					if (agent.current_pos[0] + 1, agent.current_pos[1],
					    agent.current_pos[2]) == d.location and d.is_valid(d.num_blocks):
						agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1],
						                     agent.current_pos[2])
						agent.reward += 14
						agent.have_block = 0
						d.num_blocks += 1
						agent2.other_pos = agent.current_pos
						return direction

		elif direction == 2:
			if agent.have_block == 0:
				for p in pickupArray:
					if (agent.current_pos[0], agent.current_pos[1] - 1,
					    agent.current_pos[2]) == p.location and p.is_valid(p.num_blocks):
						agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1,
						                     agent.current_pos[2])
						agent.reward += 14
						agent.have_block = 1
						p.num_blocks -= 1
						agent2.other_pos = agent.current_pos
						return direction
			else:
				for d in dropoffArray:
					if (agent.current_pos[0], agent.current_pos[1] - 1,
					    agent.current_pos[2]) == d.location and d.is_valid(d.num_blocks):
						agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1,
						                     agent.current_pos[2])
						agent.reward += 14
						agent.have_block = 0
						d.num_blocks += 1
						agent2.other_pos = agent.current_pos
						return direction

		elif direction == 3:
			if agent.have_block == 0:
				for p in pickupArray:
					if (agent.current_pos[0], agent.current_pos[1] + 1,
					    agent.current_pos[2]) == p.location and p.is_valid(p.num_blocks):
						agent.current_pos = (agent.current_pos[0], agent.current_pos[1] + 1,
						                     agent.current_pos[2])
						agent.reward += 14
						agent.have_block = 1
						p.num_blocks -= 1
						agent2.other_pos = agent.current_pos
						return direction
			else:
				for d in dropoffArray:
					if (agent.current_pos[0], agent.current_pos[1] + 1,
					    agent.current_pos[2]) == d.location and d.is_valid(d.num_blocks):
						agent.current_pos = (agent.current_pos[0], agent.current_pos[1] + 1,
						                     agent.current_pos[2])
						agent.reward += 14
						agent.have_block = 0
						d.num_blocks += 1
						agent2.other_pos = agent.current_pos
						return direction

		elif direction == 4:
			if agent.have_block == 0:
				for p in pickupArray:
					if (agent.current_pos[0], agent.current_pos[1],
					    agent.current_pos[2] + 1) == p.location and p.is_valid(p.num_blocks):
						agent.current_pos = (agent.current_pos[0], agent.current_pos[1],
						                     agent.current_pos[2] + 1)
						agent.reward += 14
						agent.have_block = 1
						p.num_blocks -= 1
						agent2.other_pos = agent.current_pos
						return direction
			else:
				for d in dropoffArray:
					if (agent.current_pos[0], agent.current_pos[1],
					    agent.current_pos[2] + 1) == d.location and d.is_valid(d.num_blocks):
						agent.current_pos = (agent.current_pos[0], agent.current_pos[1],
						                     agent.current_pos[2] + 1)
						agent.reward += 14
						agent.have_block = 0
						d.num_blocks += 1
						agent2.other_pos = agent.current_pos
						return direction
		elif direction == 5:
			if agent.have_block == 0:
				for p in pickupArray:
					if (agent.current_pos[0], agent.current_pos[1],
					    agent.current_pos[2] - 1) == p.location and p.is_valid(p.num_blocks):
						agent.current_pos = (agent.current_pos[0], agent.current_pos[1],
						                     agent.current_pos[2] - 1)
						agent.reward += 14
						agent.have_block = 1
						p.num_blocks -= 1
						agent2.other_pos = agent.current_pos
						return direction
			else:
				for d in dropoffArray:
					if (agent.current_pos[0], agent.current_pos[1],
					    agent.current_pos[2] - 1) == d.location and d.is_valid(d.num_blocks):
						agent.current_pos = (agent.current_pos[0], agent.current_pos[1],
						                     agent.current_pos[2] - 1)
						agent.reward += 14
						agent.have_block = 0
						d.num_blocks += 1
						agent2.other_pos = agent.current_pos
						return direction

		return -1

	def takeDirection(self, agent, agent2, world, direction):
		if direction == 0:
			agent.current_pos = (agent.current_pos[0] - 1, agent.current_pos[1],
			                     agent.current_pos[2])
			agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][
			 agent.current_pos[2]]
			agent2.other_pos = agent.current_pos
		elif direction == 1:
			agent.current_pos = (agent.current_pos[0] + 1, agent.current_pos[1],
			                     agent.current_pos[2])
			agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][
			 agent.current_pos[2]]
			agent2.other_pos = agent.current_pos
		elif direction == 2:
			agent.current_pos = (agent.current_pos[0], agent.current_pos[1] - 1,
			                     agent.current_pos[2])
			agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][
			 agent.current_pos[2]]
			agent2.other_pos = agent.current_pos
		elif direction == 3:
			agent.current_pos = (agent.current_pos[0], agent.current_pos[1] + 1,
			                     agent.current_pos[2])
			agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][
			 agent.current_pos[2]]
			agent2.other_pos = agent.current_pos
		elif direction == 4:
			agent.current_pos = (agent.current_pos[0], agent.current_pos[1],
			                     agent.current_pos[2] + 1)
			agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][
			 agent.current_pos[2]]
			agent2.other_pos = agent.current_pos
		elif direction == 5:
			agent.current_pos = (agent.current_pos[0], agent.current_pos[1],
			                     agent.current_pos[2] - 1)
			agent.reward += world[agent.current_pos[0]][agent.current_pos[1]][
			 agent.current_pos[2]]
			agent2.other_pos = agent.current_pos

	def PRandom(
	  self, agent, agent2, world, pickupArray,
	  dropoffArray):  # returns the direction that the policy decided to take
		directions = self.directionParser(agent)
		for direction in directions:  # this is to check if there is a pick up or drop off available
			if self.pickUpAndDropOffCheck(agent, agent2, direction, pickupArray,
			                              dropoffArray) != -1:
				return direction
		r = np.random.choice(directions)
		self.takeDirection(agent, agent2, world, r)
		return r

	def PGreedy(self, agent, agent2, world, pickupArray, dropoffArray):
		directions = self.directionParser(agent)
		print("This are the directions: ", directions)
		for direction in directions:  # this is to check if there is a pick up or drop off available
			if self.pickUpAndDropOffCheck(agent, agent2, direction, pickupArray,
			                              dropoffArray) != -1:
				print("I found a reward")
				return

	def PExploit(self, agent, agent2, world, pickupArray, dropoffArray):
		directions = self.directionParser(agent)
		print("This are the directions: ", directions)
		for direction in directions:  # this is to check if there is a pick up or drop off available
			if self.pickUpAndDropOffCheck(agent, agent2, direction, pickupArray,
			                              dropoffArray) != -1:
				print("I found a reward")
				return


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


class Qtable:
	Qtable = np.zeros((3, 3, 3, 2, 6))  # always starts off at zero

	# Q[Z][Y][X][block or no block][action]  <-- This is how the q-table is set up

	# formula Q(S_new, a) = Q(S_old, a) + alpha(reward + gamma * max(q-value of all states given the action) - Q(S_old, a))
	# explore first (that's where the 500 steps come in)
	# hence why we call prandom because it randomly decides what future state to go to

	def qLearning(self, m_agent, f_agent, world, var_gamma, var_alpha, num_steps,
	              pickupArray, dropoffArray):
		p = Policy()  # <--- This lets us access the policies
		q = Qtable()  # <--- This lets us access the qtable
		for i in range(num_steps):  # <--- here we iterate through the number of steps per experitment
			old_state_m = m_agent.current_pos
			old_state_f = f_agent.current_pos  # <--- memorize what the past

			m = p.PRandom(m_agent, f_agent, world, pickupArray,
			              dropoffArray)  # running the policy for the male agent
			f = p.PRandom(f_agent, m_agent, world, pickupArray,
			              dropoffArray)  # running the policy for the female agent
			# remember I update the rewards, have block(for agents), num_blocks(for pickup/dropoff) and positions in the policy so no need to do in this function

			new_state_m = m_agent.current_pos
			new_state_f = f_agent.current_pos

			m_future_directions = p.directionParser(m_agent)
			m_future_directions_qvalue = []
			for direction in m_future_directions:
				if m_agent.have_block == 0:
					m_future_directions_qvalue.append(
					 q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][0][direction])
				elif m_agent.have_block == 1:
					m_future_directions_qvalue.append(
					 q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][1][direction])

			f_future_directions = p.directionParser(f_agent)
			f_future_directions_qvalue = []
			for direction in f_future_directions:
				if f_agent.have_block == 0:
					f_future_directions_qvalue.append(
					 q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][0][direction])
				elif f_agent.have_block == 1:
					f_future_directions_qvalue.append(
					 q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][1][direction])

			if m_agent.have_block == 0:
				q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][0][m] = q.Qtable[
				 old_state_m[0]][old_state_m[1]][old_state_m[2]][0][m] + var_alpha * (
				  world[new_state_m[0]][new_state_m[1]][new_state_m[2]] +
				  var_gamma * max(m_future_directions_qvalue) -
				  q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][0][m])
			elif m_agent.have_block == 1:
				q.Qtable[new_state_m[0]][new_state_m[1]][new_state_m[2]][1][m] = q.Qtable[
				 old_state_m[0]][old_state_m[1]][old_state_m[2]][1][m] + var_alpha * (
				  world[new_state_m[0]][new_state_m[1]][new_state_m[2]] +
				  var_gamma * max(m_future_directions_qvalue) -
				  q.Qtable[old_state_m[0]][old_state_m[1]][old_state_m[2]][1][m])

			# go to the beginning of the qtable class
			if f_agent.have_block == 0:
				q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][0][f] = q.Qtable[
				 old_state_f[0]][old_state_f[1]][old_state_f[2]][0][f] + var_alpha * (
				  world[new_state_f[0]][new_state_f[1]][new_state_f[2]] +
				  var_gamma * max(f_future_directions_qvalue) -
				  q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][0][f])
			elif f_agent.have_block == 1:
				q.Qtable[new_state_f[0]][new_state_f[1]][new_state_f[2]][1][f] = q.Qtable[
				 old_state_f[0]][old_state_f[1]][old_state_f[2]][1][f] + var_alpha * (
				  world[new_state_f[0]][new_state_f[1]][new_state_f[2]] +
				  var_gamma * max(f_future_directions_qvalue) -
				  q.Qtable[old_state_f[0]][old_state_f[1]][old_state_f[2]][1][f])

			count = 0

			for d in dropoffArray:
          print(d.num_blocks)
      return
				# if not d.is_valid(d.num_blocks):
				# 	count += 1
				# if not d.is_valid(d.num_blocks) and count == len(dropoffArray) - 1:
				# 	return q.Qtable

		
  return q.Qtable


var_alpha = 0.3
var_gamma = 0.5

# initialized
fem_agent = Agent((0, 0, 0), (2, 1, 2), 0,
                  0)  # her pos, his pos, reward, have_block
male_agent = Agent((2, 1, 2), (0, 0, 0), 0, 0)
# cells
pickup1 = PickUp(10, (2, 2, 1),
                 14)  # number of blocks currently held, location of the cells
pickup2 = PickUp(10, (1, 1, 0), 14)
dropoff1 = DropOff(0, (0, 0, 2), 14)
dropoff2 = DropOff(0, (2, 1, 2), 14)
dropoff3 = DropOff(0, (0, 0, 1), 14)
dropoff4 = DropOff(0, (2, 0, 0), 14)
#risky1 = c.Risky(0, )
pickupArray = [pickup1, pickup2]
dropoffArray = [dropoff1, dropoff2, dropoff3, dropoff4]

world = [
 [[-1, -1, -1], [-1, 14, -1], [14, -2, -1]],  #1
 [[14, -1, -1], [-1, -2, -1], [-1, -1, 14]],  #2
 [[14, -1, -1], [-1, -1, -1], [-1, 14, -1]]
]  #3

numSteps = 10000

q = Qtable()
print(
 "Q-Table: ",
 q.qLearning(male_agent, fem_agent, world, var_gamma, var_alpha, numSteps,
             pickupArray, dropoffArray))
print("Male agent position: ", male_agent.current_pos,
      " Female agent position: ", fem_agent.current_pos)
for p in pickupArray:
	print("This is pickup: ", p.num_blocks)

for d in dropoffArray:
	print("This is a dropoff: ", d.num_blocks)