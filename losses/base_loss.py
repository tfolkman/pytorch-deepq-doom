from abc import ABC
from utils.helper_functions import get_device


class AbstractLoss(ABC):

    def __init__(self, model, gamma, optim):
        """
        :param gamma: value for gamma
        :param optim: the optimizer
        """
        super().__init__()
        self.gamma = gamma
        self.optim = optim
        self.device = get_device()
        self.model = model

