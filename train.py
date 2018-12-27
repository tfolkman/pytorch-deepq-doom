from comet_ml import Experiment
import torch
import logging
import click

from losses.td_loss import TDLoss
from models.initializers import init_weights
from utils.config_options import load_config
from utils.helper_functions import epsilon_greedy_move, handle_done, handle_not_done, initialize_memory, get_device
from memory.simple import SimpleMemory
from models.cnns import DeepQ
from games.doom_basic import DoomBasic

logging.basicConfig(level="INFO")
log = logging.getLogger(__name__)


@click.command()
@click.option('--key', help='your comet ml api key')
@click.option('--config', default='config.json', help='path to your config file. default is config.json')
def train(key, config):
    """
    Train Doom Using a DQN
    """
    config_options = load_config(config)

    experiment = Experiment(api_key=key, project_name=config_options['name'],
                            disabled=config_options['disable_comet'])
    experiment.log_parameters(config_options)

    devices, more_than_one_device = get_device()

    memory = SimpleMemory(config_options["memory_size"])
    model = DeepQ().to(devices[0])
    target = DeepQ().to(devices[0])
    model.apply(init_weights)
    target.load_state_dict(model.state_dict())
    target.eval()
    optim = torch.optim.Adam(model.parameters(), config_options['lr'])
    loss_function = TDLoss(model, target, config_options['gamma'], optim)
    game = DoomBasic()
    initialize_memory(config_options["batch_size"], game, memory)

    total_steps = 0
    eps_threshold = 1.0
    game.init()
    for episode in range(config_options["total_episodes"]):
        state, game_start = game.start_new_game()
        loss_sum = 0
        reward_sum = 0
        memory.reset_stack()
        for step in range(config_options["max_steps"]):

            state_trans = memory.transform(state)

            if step == 0:
                memory.fill_stack(state_trans)

            reward, action, done, eps_threshold = epsilon_greedy_move(game, model, memory.get_stacked_states(),
                                                                      config_options, total_steps)
            reward_sum += reward
            total_steps += 1
            if done:
                handle_done(memory.get_stacked_states(), action, reward,  memory)
                break
            else:
                state, game_start = handle_not_done(game, memory.get_stacked_states(), action, reward,  memory)

            loss_sum += loss_function.update_model(memory.sample(config_options["batch_size"]))

        if episode % config_options["update_target_every"] == 0:
            target.load_state_dict(model.state_dict())

        log_statistics_and_save(loss_sum, reward_sum, eps_threshold, config_options['max_steps'], episode,
                                config_options['save_every'], config_options['save_file'],
                                model, experiment)


def log_statistics_and_save(loss_sum, reward_sum, eps_threshold, steps, episode, save_every, save_file,
                            model, experiment):
    avg_loss = loss_sum / steps
    experiment.log_metrics({"reward_total": reward_sum, "avg_loss": avg_loss, "eps": eps_threshold}, step=episode)
    if episode % save_every == 0:
        log.info("Avg Loss: {}".format(avg_loss))
        log.info("Total Reward: {}".format(reward_sum))
        log.info("Explore Prob: {}".format(eps_threshold))
        log.info("Epoch: {}".format(episode))
        log.info("Saving...")
        torch.save(model.state_dict(), save_file)


if __name__ == "__main__":
    train()
