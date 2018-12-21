import math
import torch
import random
import numpy as np


def handle_done(transformed_state, action, reward,  memory):
    next_state = torch.zeros_like(transformed_state)
    memory.push(transformed_state, action, reward, next_state, False)


def handle_not_done(game, state_trans, action, reward, memory):
    next_state = game.get_state()
    next_state_trans = memory.transform(next_state)
    memory.push(state_trans, action, reward, next_state_trans, True)
    return next_state, False


def initialize_memory(pretrain_length, game,  memory):
    state, game_start = game.start_new_game()

    for i in range(pretrain_length):

        state_trans = memory.transform(state)

        # Random action
        action = random.choice(game.possible_actions)
        reward, done = game.take_action(action)

        # If we're dead
        if done:
            handle_done(state_trans, action, reward, memory)
            state, game_start = game.start_new_game()

        else:
            state, game_start = handle_not_done(game, state_trans, action, reward, memory)


def epsilon_greedy_move(game, model, state, config, steps_done):
    eps_threshold = config["explore_stop"] + (config["explore_start"] - config["explore_stop"]) \
        * math.exp(-1. * steps_done / config["decay"])
    if np.random.rand() > eps_threshold:
        with torch.no_grad():
            action = game.possible_actions[int(torch.argmax(model(state)))]
    else:
        action = random.choice(game.possible_actions)
    reward, done = game.take_action(action)
    return reward, action, done, eps_threshold


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
