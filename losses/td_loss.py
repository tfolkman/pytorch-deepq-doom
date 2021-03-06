import torch

from losses.base_loss import AbstractLoss


class TDLoss(AbstractLoss):

    def update_model(self, memory_sample):
        """
        Updates the model with a smooth l1 loss and gradient clipping
        :param memory_sample: Assumes of type MemoryItem (see base_memory.py)
        :return:
        """
        best_action_next_state = torch.argmax(self.model(memory_sample.next_state).detach(), 1).unsqueeze(-1)
        q_next_state = self.target_model(memory_sample.next_state).detach()
        best_q_next_state = q_next_state.gather(1, best_action_next_state).squeeze(-1)
        target = memory_sample.reward + (self.gamma * best_q_next_state * memory_sample.not_done)
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
