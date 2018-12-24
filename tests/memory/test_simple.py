import torch

from memory.simple import SimpleMemory
import numpy as np

from utils.helper_functions import get_device

memory = SimpleMemory(100)
device = get_device()


def create_random_trans_tensor():
    return memory.transform(np.random.rand(120, 160))


def test_fill_stack():
    rnd_tensor = create_random_trans_tensor()
    memory.fill_stack(rnd_tensor)
    for i in range(4):
        assert torch.all(torch.eq(memory.stacked_frames[i], rnd_tensor))


def test_reset_stack():
    rnd_tensor = create_random_trans_tensor()
    memory.fill_stack(rnd_tensor)
    memory.reset_stack()
    for i in range(4):
        assert torch.all(torch.eq(memory.stacked_frames[i], torch.zeros_like(memory.stacked_frames[i])))


def test_append_to_stack():
    memory.reset_stack()
    for i in range(5):
        rnd_tensor = create_random_trans_tensor()
        memory.append_to_stack(rnd_tensor)
        assert torch.all(torch.eq(memory.stacked_frames[-1], rnd_tensor))


def test_get_stacked_states():
    rnd_tensor_all = torch.zeros(1, 4, 84, 84).to(device)
    for i in range(4):
        rnd_tensor = create_random_trans_tensor()
        rnd_tensor_all[:, i, :, :] = rnd_tensor
        memory.append_to_stack(rnd_tensor)
    stacked_states = memory.get_stacked_states()
    assert torch.all(torch.eq(stacked_states, rnd_tensor_all))
