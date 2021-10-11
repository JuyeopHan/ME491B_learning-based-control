import numpy as np
import copy
from environment import GridWorld


class Agent:
    def __init__(self, environment, discount_factor=0.99, epsilon=0.2, lamb=0.8, learning_rate=0.01):
        self.env = environment
        self.actions = [0, 1, 2, 3]

        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.lamb = lamb
        self.learning_rate = learning_rate

        ################

        # your code
        self.annealing_factor = 0.99999
        self.q_table = np.zeros([self.env.size[0], self.env.size[1], len(self.actions)])
        self.policy = np.zeros([self.env.size[0], self.env.size[1]])
        self.state_sequence = []
        self.action_sequence = []
        self.reward_sequence = []
        self.next_state_sequence = []
        self.next_action_sequence = []
        #################

    def reset(self):
        # reset the agent each episode
        ################

        # your code
        self.epsilon = self.annealing_factor * self.epsilon
        self.state_sequence = []
        self.action_sequence = []
        self.reward_sequence = []
        self.next_state_sequence = []
        self.next_action_sequence = []
        #################

    def act(self, state):
        # sample an available action
        ################
        # your code
        accum_prob = 0
        policy_accum_prob_pair_table = [["start",0]]
        q_values ={}
        # getting a possible actions
        actions = self.env.get_possible_actions(state)
        len_actions = len(actions)

        # get a argmax action
        for action in actions:
            q_values[action] = self.q_table[state[0],state[1],action]
        argmax_action = max(q_values, key= q_values.get)

        # preprocessing for epsilon greedy
        for action in actions:
            if action == argmax_action:
                accum_prob = accum_prob + (1 - self.epsilon)
            else:
                accum_prob = accum_prob + self.epsilon/(len_actions-1)
            action_prob_pair = [copy.deepcopy(action), copy.deepcopy(accum_prob)]
            policy_accum_prob_pair_table.append(action_prob_pair)

        # epsilon greedy
        rand_sample = np.random.random()
        action = -1
        for i in range(len(policy_accum_prob_pair_table)-1):
            
            curr_acumm_prob = policy_accum_prob_pair_table[i][1]
            next_accmm_prob = policy_accum_prob_pair_table[i+1][1]
            curr_action = policy_accum_prob_pair_table[i+1][0]
            if (curr_acumm_prob <= rand_sample) and (rand_sample < next_accmm_prob):
                action = curr_action
                break
        #################
        return action

    def update_policy(self):
        # make greedy policy w.r.t. the value function
        ################

        # your code
        for row in range(self.env.size[0]):
            for col in range(self.env.size[1]):
                max_q_value = -1e6
                max_policy = -1
                actions = self.env.get_possible_actions([row,col])
                for action in actions:
                    if(max_q_value < self.q_table[row,col,action]):
                        max_q_value = self.q_table[row,col,action]
                        max_policy = action
                self.policy[row,col] = max_policy
        
        self.policy = self.policy.astype(int)

    def store_transition(self, state, action, reward, next_state):
        # store the transition for offline update
        ################

        # your code
        self.state_sequence.append(copy.deepcopy(state))
        self.action_sequence.append(copy.deepcopy(action))
        self.reward_sequence.append(copy.deepcopy(reward))
        self.next_state_sequence.append(copy.deepcopy(next_state))
        if next_state is not self.env.goal:
            next_action = self.act(next_state)
            self.next_action_sequence.append(copy.deepcopy(next_action))
        else:
            next_action = 0
            self.next_action_sequence.append(copy.deepcopy(next_action))
        return next_action
            
        #################

    def update(self):
        # update the value function
        ################

        # your code
        sample_length = len(self.state_sequence)
        # for each node at sample paths
        for idx in range(sample_length):
            state = self.state_sequence[idx]
            action = self.action_sequence[idx]
            
            q_sum = 0
            reward_sum = 0

            td_length = sample_length - idx

            # getting a q_lambda
            for td_step in range(td_length - 1):
                q_lambda = 0
                reward = self.reward_sequence[idx + td_step]
                next_state = self.next_state_sequence[idx + td_step]
                next_action = self.next_action_sequence[idx + td_step]

                reward_sum = reward_sum + (self.discount_factor)**td_step * reward
                q_lambda = reward_sum + (self.discount_factor)**(td_step + 1) * self.q_table[next_state[0], next_state[1], next_action]
                q_lambda = self.lamb**td_step * q_lambda
                q_sum = q_sum + q_lambda
            
            reward = self.reward_sequence[sample_length - 1]
            next_state = self.next_state_sequence[sample_length - 1]
            next_action = self.next_action_sequence[sample_length - 1]
            reward_sum = reward_sum + (self.discount_factor)**(td_length - 1) * reward
            q_lambda = reward_sum + (self.discount_factor)**td_length * self.q_table[next_state[0], next_state[1], next_action]
            q_lambda = self.lamb**(td_length - 1) * q_lambda

            q_sum = (1 - self.lamb)*q_sum + q_lambda

            # update q_values
            self.q_table[state[0], state[1], action] = self.q_table[state[0], state[1], action] + self.learning_rate*(q_sum - self.q_table[state[0], state[1], action])

        #################


if __name__ == '__main__':
    env = GridWorld()
    agent = Agent(environment=env)

    for episode in range(1000000):

        # uniformly sample the initial state. reject the terminal state
        ################
        # your code
        #sampling state
        terminal_num = env.goal[0]*env.size[0] + env.goal[1]
        while True:
            state_num = np.random.randint(low = 0, high= env.size[0]*env.size[1] -1, size = 1)
            if (state_num != terminal_num):
                init_element1 = state_num[0] // env.size[0]
                init_element2 = state_num[0] % env.size[0]
                init_state = [init_element1, init_element2]
                break
        
        #sampling action
        actions = agent.env.get_possible_actions(init_state)
        while True:
            action = np.random.randint(low = 0, high= 3, size = 1)
            action = action[0]
            if (action in actions):
                break

        #################

        state = env.reset(init_state)
        agent.reset()
        steps = 0

        # actual training
        while True:
            ################
            # your code
            next_state, reward, done = env.step(action)
            next_action = agent.store_transition(state, action, reward, next_state)
            if done is True:
                agent.update()
            state = next_state
            action = next_action
            steps = steps + 1
            #################
            if done or steps >= 1000:
                break
        agent.update()

        if episode % 10000 == 0:
            agent.update_policy()
            print("episode : {}, steps : {}".format(episode, steps))

    agent.update_policy()
    np.save("prob3_sarsa_forward_policy.npy", agent.policy)
    np.save("prob3_sarsa_forward_q_table.npy", agent.q_table)
