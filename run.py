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
@click.option('--weights', default='./model_weights/doom_dqn.state',
              help='path to your trained weights. default is ./model_weights/doom_dqn.state')
@click.option('--n', default=5, help="number of games to play. default 5.")
@click.option('--sleep', default=.02, help="time to sleep between each move. default is .02")
def play_game(weights, n, sleep):
    game = DoomBasic()
    game.set_window_visibility(True)
    device = get_device()
    model = DeepQ(1.0).to(device)
    model.load_state_dict(torch.load(weights))
    memory = SimpleMemory(1)
    for i in range(n):
        _, _ = game.start_new_game()
        total_reward = 0
        state = memory.transform(game.get_state())
        while not game.is_done():
            predictions = model(state)
            action = game.possible_actions[int(torch.argmax(predictions))]
            reward, _ = game.take_action(action)
            total_reward += reward
            if game.is_done():
                break
            state = memory.transform(game.get_state())
            time.sleep(sleep)
        log.info("Total reward for game {0}: {1}".format(i, total_reward))
    game.close()


if __name__ == "__main__":
    play_game()
