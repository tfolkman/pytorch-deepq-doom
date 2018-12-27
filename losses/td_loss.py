import torch

from losses.base_loss import AbstractLoss


class TDLoss(AbstractLoss):

    def update_model(self, memory_sample):
        """
        Updates the model with a smooth l1 loss and gradient clipping
        :param memory_sample: Assumes of type MemoryItem (see base_memory.py)
        :return:
        """
        max_q_next_state, _ = torch.max(self.target_model(memory_sample.next_state).detach(), 1)
        target = memory_sample.reward + (self.gamma * max_q_next_state * memory_sample.not_done)
        action_indexes = torch.argmax(memory_sample.action, 1).unsqueeze(1)
        action_indexes = action_indexes.to(self.device)
        predicted = self.model(memory_sample.state).gather(1, action_indexes).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        self.optim.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        return loss.item()
