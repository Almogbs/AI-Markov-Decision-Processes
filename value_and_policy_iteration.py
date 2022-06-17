from copy import deepcopy
#from plistlib import UID
#from tkinter import N
import numpy as np
from numpy import vsplit
from mdp import *


NUM_TO_ACTION = {
    0: 'UP',
    1: 'DOWN',
    2: 'RIGHT',
    3: 'LEFT'
}

def get_neighbors(i, j, mdp: MDP):
    neighbors = {}
    for action, value in mdp.actions.items():
        new_i, new_j = i + value[0], j + value[1]
        if  new_i < mdp.num_row and new_j < mdp.num_col         \
            and new_i >= 0 and new_j >= 0 and mdp.board[new_i][new_j] != 'WALL':
            neighbors[action] = (new_i, new_j)

    return neighbors 

def get_arg_of_max(i, j, mdp: MDP, U: List):
    arg_of_max = -np.inf
    action_of_max = None
    for action, state in get_neighbors(i, j, mdp):
        if U[state[0], state[1]] > arg_of_max:
            arg_of_max = mdp.transition_function[action]*U[state[0], state[1]]
            action_of_max = action
    
    return (action_of_max, arg_of_max)


def value_iteration(mdp: MDP, U_init: List, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #
    # ====== CODE: ======
    new_U = deepcopy(U_init)
    while True:
        curr_U = new_U
        lamda = 0
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                state = (i, j)

                if mdp.board[i][j] == 'WALL':
                    continue
                elif state in mdp.terminal_states:
                    state_reward = int(mdp.board[i][j])
                    continue

                state_reward = int(mdp.board[i][j])
                next_states = {}
                for action in mdp.actions.keys():
                    next_states[action] = mdp.step(state, action)
                
                sums = []
                for action in next_states.keys():
                    sum = 0
                    for idx, prob in enumerate(mdp.transition_function[action]):
                        actual_new_state = (i, j) + mdp.actions[NUM_TO_ACTION[idx]]
                        sum += prob * curr_U[actual_new_state[0]][actual_new_state[1]]
                    sums.append(sum)
                
                new_U[i][j] = state_reward + mdp.gamma * max(sums)
                if abs(new_U[i][j] - curr_U[i][j]) > lamda:
                    lamda = new_U[i][j]

        if lamda < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break
        
    return curr_U
    # ========================



def get_policy(mdp: MDP, U: List) -> List:
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    # ====== CODE: ======
    policy = []
    for _ in range(mdp.num_row):
        policy.append([0] * mdp.num_col)


    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            state = (i, j)

            if mdp.board[i][j] == 'WALL':
                policy[i][j] = 'WALL'
            elif state in mdp.terminal_states:
                policy[i][j] = 0
            else:
                neighbors = get_neighbors(i, j, mdp)
                actions = list(neighbors.keys())
                rewards = [U[n[0]][n[1]] for n in neighbors.values()]

                max_reward = max(rewards)
                policy[i][j] = actions[rewards.index(max_reward)]
    
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
