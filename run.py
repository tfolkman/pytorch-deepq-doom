import logging
import time
import torch
import click
from games.doom_basic import DoomBasic
from memory.simple import SimpleMemory
from models.cnns import DeepQ
from utils.helper_functions import get_device

logging.basicConfig(level="INFO")
log = logging.getLogger(__name__)


@click.command()
@click.option('--weights', default='./model_weights/doom_dqn_stacked.state',
              help='path to your trained weights. default is ./model_weights/doom_dqn_stacked.state')
@click.option('--n', default=10, help="number of games to play. default 10.")
@click.option('--sleep', default=.02, help="time to sleep between each move. default is .02")
def play_game(weights, n, sleep):
    game = DoomBasic()
    game.set_window_visibility(True)
    device = get_device()
    model = DeepQ().to(device)
    model.load_state_dict(torch.load(weights))
    memory = SimpleMemory(1)
    for i in range(n):
        _, _ = game.start_new_game()
        total_reward = 0
        state = memory.transform(game.get_state())
        memory.fill_stack(state)
        while not game.is_done():
            predictions = model(memory.get_stacked_states().to(get_device()))
            action = game.possible_actions[int(torch.argmax(predictions))]
            reward, _ = game.take_action(action)
            total_reward += reward
            if game.is_done():
                break
            memory.append_to_stack(memory.transform(game.get_state()))
            time.sleep(sleep)
        log.info("Total reward for game {0}: {1}".format(i, total_reward))
    game.close()


if __name__ == "__main__":
    play_game()
