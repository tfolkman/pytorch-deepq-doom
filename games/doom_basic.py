from vizdoom.vizdoom import DoomGame
from games.base_game import AbstractGame


class DoomBasic(AbstractGame):
    def __setup_game(self):
        self.game = DoomGame()
        self.game.load_config("basic.cfg")
        self.game.set_doom_scenario_path("basic.wad")

    def __initialize(self):
        self.__setup_game()
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        self.possible_actions = [left, right, shoot]

    def set_window_visibility(self, visibility):
        self.__setup_game()
        self.game.set_window_visible(visibility)

    def start_new_game(self):
        self.game.new_episode()
        game_start = True
        state = self.game.get_state().screen_buffer
        return state, game_start

    def take_action(self, action):
        return self.game.make_action(action), self.game.is_episode_finished()

    def get_next_state(self):
        return self.game.get_state().screen_buffer
