import numpy as np
import copy
from environment import GridWorld


class Agent:
    def __init__(self, environment, discount_factor=0.99, epsilon=0.2, learning_rate=0.01):
        self.env = environment
        self.actions = [0, 1, 2, 3]
        self.q_table = np.zeros(tuple(self.env.size) + (len(self.actions),))
        self.policy = np.zeros(tuple(self.env.size))

        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        self.s_state = []
        self.s_action = []
        self.s_reward = []
        self.s_next_state = []
        self.s_next_q = []

        for i in range(env.size[0]):
            for j in range(env.size[1]):
                possible_actions = env.get_possible_actions([i, j])
                self.q_table[tuple([i, j])][list(set(self.actions) - set(possible_actions))] = -1e3

    def reset(self):
        self.epsilon *= 0.99999
        self.s_state.clear()
        self.s_action.clear()
        self.s_reward.clear()
        self.s_next_state.clear()
        self.s_next_q.clear()
        #################

    def act(self, state):
        possible_actions = env.get_possible_actions(state)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(possible_actions)
        else:
            action = np.argmax(self.q_table[tuple(state)])
        return action

    def update_policy(self):
        self.policy = np.argmax(self.q_table, axis=2)

    def store_transition(self, state, action, reward, next_state):
        next_action = self.act(next_state)

        self.s_state.append(state)
        self.s_action.append(action)
        self.s_reward.append(reward)
        self.s_next_state.append(next_state)
        self.s_next_q.append(self.q_table[tuple(next_state)][next_action])

    def update(self):
        for t in range(len(self.s_state)):
            reward = self.s_reward[t]
            next_state = self.s_next_state[t]
            next_argmax_action = np.argmax(self.q_table[tuple(next_state)])
            next_q_max = self.q_table[tuple(next_state)][next_argmax_action]
            q_return = reward + self.discount_factor * next_q_max

            self.q_table[tuple(self.s_state[t])][self.s_action[t]] += \
                self.learning_rate * (q_return - self.q_table[tuple(self.s_state[t])][self.s_action[t]])


if __name__ == '__main__':
    env = GridWorld()
    agent = Agent(environment=env)

    for episode in range(1000000):

        while True:
            init_state = [np.random.choice(env.size[0]), np.random.choice(env.size[1])]
            if not init_state == env.goal:
                break

        state = env.reset(init_state)
        agent.reset()
        steps = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state)
            state = next_state
            steps += 1
            if done or steps >= 1000:
                break
        agent.update()

        if episode % 10000 == 0:
            agent.update_policy()
            print("episode : {}, steps : {}".format(episode, steps))
            for i in range(6):
                print(agent.policy[i,:])

    agent.update_policy()
    np.save("prob4_qlearning_policy.npy", agent.policy)
    np.save("prob4_qlearning_q_table.npy", agent.q_table)
