from copy import deepcopy
#from plistlib import UID
#from tkinter import N
import numpy as np
from numpy import vsplit
from mdp import *
from time import sleep


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


def value_iteration(mdp: MDP, U_init: List, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #
    # ====== CODE: ======
    new_U = deepcopy(U_init)
    while True:
        curr_U = deepcopy(new_U)
        lamda = 0
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                state = (i, j)

                if mdp.board[i][j] == 'WALL':
                    continue
                elif state in mdp.terminal_states:
                    new_U[i][j] = int(mdp.board[i][j])
                else:
                    state_reward = int(mdp.board[i][j])
                    
                    sums = []
                    for action in mdp.actions.keys():
                        sum = 0
                        for action_idx, prob in enumerate(mdp.transition_function[action]):
                            actual_new_state = mdp.step(state, NUM_TO_ACTION[action_idx])
                            sum += prob * curr_U[actual_new_state[0]][actual_new_state[1]]
                        sums.append(sum)
                    new_U[i][j] = state_reward + mdp.gamma * max(sums)
                
                if abs(new_U[i][j] - curr_U[i][j]) > lamda:

                    lamda = abs(new_U[i][j] - curr_U[i][j])

        if lamda <= epsilon * (1 - mdp.gamma) / mdp.gamma:
            break


    return curr_U
    # ========================



def get_policy(mdp: MDP, U: List) -> List:
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
    P = [[0 for _ in range(mdp.num_col * mdp.num_row)] for _ in range(mdp.num_col * mdp.num_row)]
    I = np.ones_like(P)
    R = [0 for _ in range(mdp.num_col * mdp.num_row)]
    for i in range(mdp.num_col):
        for j in range(mdp.num_row):
            curr_state = (i, j)
            R[curr_state[0]*mdp.num_col+curr_state[1]] = mdp.board[i][j]

            if curr_state not in mdp.terminal_states and mdp.board[i][j] != 'WALL':
                curr_action = policy[curr_state[0]][curr_state[1]]
                for action_prob, (action, _) in zip(mdp.transition_function[curr_action], mdp.actions):
                    print(f'curr_stat: {curr_state}, curr_action: {curr_action}, -- {mdp.actions[curr_action]}')
                    new_state = mdp.step(curr_state, action)
                    P[curr_state[0]*mdp.num_col+curr_state[1]][new_state[0]*mdp.num_col+new_state[1]] += action_prob

    res = np.linalg.inv(np.array(I - mdp.gamma*P)) @ np.array(R)

    return list(res)
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    # U = [[0 for _ in range(mdp.num_col)] for _ in range(MDP.num_row)]
    changed = True
    policy = policy_init

    while changed:
        changed = False
        util = policy_evaluation(mdp, policy)
        for i in range(mdp.num_col):
            for j in range(mdp.num_row):
                curr_state = (i, j)
                if curr_state not in mdp.terminal_states and mdp.board[i][j] != 'WALL':
                    curr_action = policy[curr_state[0]][curr_state[1]]
                    max_val = calc(mdp, curr_action, util, curr_state)
                    for action_prob, (action, _) in zip(mdp.transition_function[curr_action], mdp.actions):
                        curr_val = calc(mdp, action, util, curr_state)
                        if curr_val > max_val:
                            max_val = curr_val
                            changed = True
                            policy[curr_state[0]][curr_state[1]] = action

    return policy
    # ========================

def calc(mdp, action, util, state):
    res = 0
    for action_prob, (action, _) in zip(mdp.transition_function[action], mdp.actions):
        next_stat = mdp.step(state, action)
        res += action_prob * util[next_stat[0]][next_stat[1]]

    return res