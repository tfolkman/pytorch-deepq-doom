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
    def get_next_state(self):
        """
        Gets the next state
        :return: Next state
        """
        pass

    @abstractmethod
    def init(self):
        """
        Initializes a game
        :return:
        """
        pass