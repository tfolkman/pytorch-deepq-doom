from abc import ABC, abstractmethod
from collections import deque, namedtuple
from utils.helper_functions import get_device


class AbstractMemory(ABC):

    def __init__(self, memory_size):
        super().__init__()
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.MemoryItem = namedtuple('MemoryItem', ['state', 'action', 'reward', 'next_state', 'not_done'])
        devices, _ = get_device()
        self.device = devices[0]

    def __len__(self):
        return len(self.memory)

    @abstractmethod
    def transform(self):
        """
        Defines any transformations which must be applied to input data
        :return:
        """
        pass

    @abstractmethod
    def push(self):
        """
        Allows the user to push new values to the memory queue
        :return:
        """
        pass

    @abstractmethod
    def sample(self, batch_size):
        """
        Allows the user to sample from the memory
        :param batch_size: The size of the sample
        :return:
        """
        pass
