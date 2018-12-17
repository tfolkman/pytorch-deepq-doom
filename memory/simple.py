import PIL
import torch
import numpy as np
from torchvision import transforms

from memory.base_memory import AbstractMemory


class SimpleMemory(AbstractMemory):
    """
    Class that handles the replay buffer and takes in the raw numpy and transforms it as necessary
    Transforms:
        Crop out the unnecessary parts of image
        Normalize to 0-1
        Resize to 84x84
    """
    def __init__(self, memory_size):
        super.__init__(memory_size)
        self.img_transforms = transforms.Compose([transforms.Resize((84,84)), transforms.ToTensor()])

    def _combine_memories(self, memories):
        states, actions, rewards, next_states, not_dones = zip(*memories)
        return self.MemoryItem(torch.cat(states).to(self.device),
                               torch.LongTensor(actions).to(self.device),
                               torch.FloatTensor(rewards).to(self.device),
                               torch.cat(next_states).to(self.device),
                               torch.FloatTensor(not_dones).to(self.device))

    def transform(self, x):
        img = PIL.Image.fromarray(x)
        img_cropped = transforms.functional.crop(img, 30, 30, 80, 100)
        return self.img_transforms(img_cropped).to(self.device).unsqueeze(0)

    def push(self, state_trans, action, reward, next_state_trans, not_done):
        self.memory.append(self.MemoryItem(state_trans, action, reward, next_state_trans, not_done))

    def sample(self, batch_size):
        indxs = np.random.choice(range(len(self.memory)), batch_size, replace=False)
        return self._combine_memories([self.memory[i] for i in indxs])
