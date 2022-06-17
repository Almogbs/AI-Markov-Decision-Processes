from copy import deepcopy
import numpy as np
from mdp import *


NUM_TO_ACTION = {
    0: 'UP',
    1: 'DOWN',
    2: 'RIGHT',
    3: 'LEFT'
}

def get_weigthed_sum(mdp, action, util, state):
    res = 0
    for action_prob, curr_action in zip(mdp.transition_function[action], mdp.actions.keys()):
        next_stat = mdp.step(state, curr_action)
        res += action_prob * util[next_stat[0]][next_stat[1]]

    return res


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
    curr_U = deepcopy(U_init)

    while True:
        delta = 0
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                state = (i, j)

                if mdp.board[i][j] == 'WALL':
                    continue
                elif state in mdp.terminal_states:
                    new_U[i][j] = float(mdp.board[i][j])
                else:
                    state_reward = float(mdp.board[i][j])
                    sums = []
                    for action in mdp.actions.keys():
                        sums.append(get_weigthed_sum(mdp, action, curr_U, state))
                    new_U[i][j] = state_reward + mdp.gamma * max(sums)
                delta = max(delta, abs(new_U[i][j] - curr_U[i][j]))

        if (mdp.gamma != 1 and delta < (epsilon * (1 - mdp.gamma)) / mdp.gamma) or (mdp.gamma == 1 and delta == 0):
            break

        curr_U = deepcopy(new_U)

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
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # ====== CODE: ======
    P = [[0 for _ in range(mdp.num_col * mdp.num_row)] for _ in range(mdp.num_col * mdp.num_row)]
    I = np.zeros_like(P)
    R = [0 for _ in range(mdp.num_col * mdp.num_row)]
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            curr_state = (i, j)
            I[curr_state[0] * (mdp.num_row + 1) + curr_state[1]][curr_state[0] * (mdp.num_row + 1) + curr_state[1]] = 1
            if mdp.board[i][j] != 'WALL':
                R[curr_state[0] * (mdp.num_row + 1) + curr_state[1]] = float(mdp.board[i][j])

            if curr_state not in mdp.terminal_states and mdp.board[i][j] != 'WALL':
                curr_action = policy[curr_state[0]][curr_state[1]]
                for action_prob, action in zip(mdp.transition_function[curr_action], mdp.actions.keys()):
                    new_state = mdp.step(curr_state, action)
                    P[curr_state[0]*(mdp.num_row + 1)+curr_state[1]][new_state[0]*(mdp.num_row + 1)+new_state[1]] += action_prob

    res = np.linalg.pinv(np.array(I - mdp.gamma*np.array(P))) @ np.array(R)
    res = [[res[i*(mdp.num_row + 1) + j] for j in range(mdp.num_col)] for i in range(mdp.num_row)]

    return list(res)
    # ========================


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== CODE: ======
    changed = True
    policy = deepcopy(policy_init)

    while changed:
        changed = False
        util = policy_evaluation(mdp, policy)
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                curr_state = (i, j)
                if curr_state not in mdp.terminal_states and mdp.board[i][j] != 'WALL':
                    curr_action = policy[curr_state[0]][curr_state[1]]
                    max_val = get_weigthed_sum(mdp, curr_action, util, curr_state)
                    for action_prob, action in zip(mdp.transition_function[curr_action], mdp.actions.keys()):
                        curr_val = get_weigthed_sum(mdp, action, util, curr_state)
                        if curr_val > max_val:
                            max_val = curr_val
                            changed = True
                            policy[curr_state[0]][curr_state[1]] = action
    
    return policy
    # ========================

