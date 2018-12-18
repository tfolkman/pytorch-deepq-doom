import torch
from comet_ml import Experiment

from models.initializers import init_weights
from utils.config_options import load_config
import argparse

from utils.helper_functions import epsilon_greedy_move, handle_done, handle_not_done
from memory.simple import SimpleMemory
from models.cnns import DeepQ
from games.doom_basic import DoomBasic


def train(game, model, memory, possible_actions, config, print_every, save_file):
    total_steps = 0
    game.init()
    for episode in range(config["total_episodes"]):
        game, state, game_start = game.start_new_game()
        for step in range(config["max_steps"]):

            state_trans = memory.transform(state)

            reward, action, done, eps_threshold = epsilon_greedy_move(game, model, state_trans,
                                                                      possible_actions,
                                                                      config, total_steps)
            total_steps += 1
            if done:
                handle_done(state_trans, action, reward,  memory)
                break
            else:
                state, game_start = handle_not_done(game, state_trans, action, reward,  memory)

            loss = model.update_model(memory.sample(config["batch_size"]))
            experiment.log_metrics({"reward": reward, "loss": loss}, step=total_steps)

            if total_steps % print_every == 0:
                print("Loss: {}".format(loss))
                print("Explore Prob: {}".format(eps_threshold))
                print("Epoch: {}".format(episode))
                print("Saving...")
                torch.save(model.state_dict(), save_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Doom Using DQN")
    parser.add_argument("-k", "--key", required=True, help='comet api key')
    parser.add_argument("-c", "--config", required=True, help='path to config file')
    args = vars(parser.parse_args())

    config_options = load_config(args['config'])

    experiment = Experiment(api_key=args['key'], project_name=config_options['name'],
                            disabled=config_options['track_with_comet'])
    experiment.log_parameters(config_options)

    memory = SimpleMemory(config_options["memory_size"])
    model = DeepQ(config_options['lr'], config_options['gamma'])
    model.apply(init_weights)
    game = DoomBasic()
#    initalize_memory(pretrain_length, game, possible_actions, memory)
 #   train(game, model, memory, possible_actions, train_params)
