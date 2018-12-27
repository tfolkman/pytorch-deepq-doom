from abc import ABC
from utils.helper_functions import get_device


class AbstractLoss(ABC):

    def __init__(self, model, target_model, gamma, optim):
        """
        :param gamma: value for gamma
        :param optim: the optimizer
        """
        super().__init__()
        self.gamma = gamma
        self.optim = optim
        devices, _ = get_device()
        self.device = devices[0]
        self.model = model
        self.target_model = target_model

