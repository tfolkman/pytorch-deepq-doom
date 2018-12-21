from abc import ABC, abstractmethod


class AbstractGame(ABC):

    def __init__(self):
        super().__init__()
        self.game = None
        self.possible_actions = None
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """
        Sets the self variables of
        possible_actions -> a list of possible moves
        game -> the game
        :return: None
        """
        pass

    @abstractmethod
    def start_new_game(self):
        """
        starts a new game
        :return: the starting state, and a flag to represent new game
        """
        pass

    @abstractmethod
    def take_action(self, action):
        """
        Takes an action for the game
        :param action:
        :return: Reward from action as well as whether game is done or not
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Gets the state
        :return: state
        """
        pass

    @abstractmethod
    def init(self):
        """
        Initializes a game
        :return:
        """
        pass

    @abstractmethod
    def is_done(self):
        """
        Returns whether game is done or not
        :return:
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the game
        :return:
        """
        pass

    @abstractmethod
    def set_window_visibility(self):
        """
        Sets whether the game window is shown
        :return:
        """
        pass
