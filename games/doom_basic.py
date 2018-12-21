from vizdoom.vizdoom import DoomGame
from games.base_game import AbstractGame
import os


class DoomBasic(AbstractGame):
    def _setup_game(self):
        self.game = DoomGame()
        self.file_path = os.path.dirname(__file__)
        self.game.load_config(os.path.join(self.file_path, "basic.cfg"))
        self.game.set_doom_scenario_path(os.path.join(self.file_path, "basic.wad"))

    def _initialize(self):
        self._setup_game()
        self.init()
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        self.possible_actions = [left, right, shoot]

    def set_window_visibility(self, visibility):
        self._setup_game()
        self.game.set_window_visible(visibility)
        self.init()

    def start_new_game(self):
        self.game.new_episode()
        game_start = True
        state = self.game.get_state().screen_buffer
        return state, game_start

    def take_action(self, action):
        return self.game.make_action(action), self.game.is_episode_finished()

    def get_state(self):
        return self.game.get_state().screen_buffer

    def init(self):
        self.game.init()

    def is_done(self):
        return self.game.is_episode_finished()

    def close(self):
        self.game.close()