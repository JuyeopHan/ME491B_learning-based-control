import numpy as np
import copy


class GridWorld:
    def __init__(self):
        self.actions = [0, 1, 2, 3]
        self.position = [0, 0]
        self.goal = [5, 5]
        self.size = [6, 6]
        self.reward = np.array([[[0.0, -3.5,  0.0, -0.5],
                                 [0.0, -2.0, -0.5, -0.5],
                                 [0.0, -0.5, -0.5, -2.0],
                                 [0.0, -1.5, -2.0, -0.5],
                                 [0.0, -3.0, -0.5, -0.5],
                                 [0.0, -4.0, -0.5,  0.0]],
                                [[-3.5, -0.5,  0.0, -2.0],
                                 [-2.0, -1.5, -2.0, -2.5],
                                 [-0.5, -0.5, -2.5, -0.5],
                                 [-1.5, -2.5, -0.5, -2.0],
                                 [-3.0, -0.5, -2.0, -0.5],
                                 [-4.0, -0.5, -0.5,  0.0]],
                                [[-0.5, -0.5,  0.0, -0.5],
                                 [-1.5, -1.0, -0.5, -1.5],
                                 [-0.5, -4.0, -1.5, -1.0],
                                 [-2.5, -2.5, -1.0, -1.5],
                                 [-0.5, -4.0, -1.5, -4.5],
                                 [-0.5, -7.5, -4.5,  0.0]],
                                [[-0.5, -1.5,  0.0, -2.5],
                                 [-1.0, -2.5, -2.5, -3.5],
                                 [-4.0, -1.5, -3.5, -1.5],
                                 [-2.5, -1.0, -1.5, -0.5],
                                 [-4.0, -0.5, -0.5, -1.0],
                                 [-7.5, -2.5, -1.0,  0.0]],
                                [[-1.5, -0.5,  0.0, -0.5],
                                 [-2.5, -1.0, -0.5, -2.0],
                                 [-1.5, -2.0, -2.0, -3.0],
                                 [-1.0, -1.0, -3.0, -4.0],
                                 [-0.5, -5.5, -4.0, -5.5],
                                 [-2.5, -1.0, -5.5,  0.0]],
                                [[-0.5,  0.0,  0.0, -4.0],
                                 [-1.0,  0.0, -4.0, -1.0],
                                 [-2.0,  0.0, -1.0, -1.0],
                                 [-1.0,  0.0, -1.0, -0.5],
                                 [-5.5,  0.0, -0.5, -0.5],
                                 [-1.0,  0.0, -0.5,  0.0]]],
                               dtype=np.float32)

    def reset(self, initial_position):
        if type(initial_position) is list and len(initial_position) == 2:
            self.position = initial_position
            return copy.deepcopy(self.position)
        else:
            raise Exception("reset failed!!!")

    def get_possible_actions(self, state):
        p_actions = [0, 1, 2, 3]

        if state[0] == 0:
            p_actions.remove(0)
        if state[0] == self.size[0] - 1:
            p_actions.remove(1)
        if state[1] == 0:
            p_actions.remove(2)
        if state[1] == self.size[1] - 1:
            p_actions.remove(3)
        return p_actions

    def step(self, action):
        if action not in self.actions:
            raise Exception("action should be a element of {0, 1, 2, 3}")

        if action not in self.get_possible_actions(self.position):
            raise Exception("cannot move outside of the world!")

        reward = self.reward[tuple(self.position)][action]

        if action == 0:
            self.position[0] -= 1
        elif action == 1:
            self.position[0] += 1
        elif action == 2:
            self.position[1] -= 1
        elif action == 3:
            self.position[1] += 1

        done = False
        if self.position == self.goal:
            done = True

        return copy.deepcopy(self.position), reward, done
