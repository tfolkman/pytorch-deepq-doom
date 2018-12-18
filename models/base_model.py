from abc import ABC, abstractmethod
import torch
from torch import nn


class AbstractModel(nn.Module, ABC):

    def __init__(self, lr, gamma):
        """
        Assumes adam optimizer with defaults except set lr
        :param lr:
        :param gamma:
        :param save_path: location to save weights
        """
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    @abstractmethod
    def forward(self, *input):
        pass

    def update_model(self, memory_sample):
        """
        Updates the model with a smooth l1 loss and gradient clipping
        :param memory_sample: Assumes of type MemoryItem (see base_memory.py)
        :return:
        """
        max_q_next_state, _ = torch.max(self(memory_sample.next_state).detach(), 1)
        target = memory_sample.reward + (self.gamma * max_q_next_state * memory_sample.not_done)
        action_indexes = torch.argmax(memory_sample.action, 1).unsqueeze(1)
        action_indexes = action_indexes.to(self.device)
        predicted = self(memory_sample.state).gather(1, action_indexes).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predicted, target)
        self.optim.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        return loss.item()
