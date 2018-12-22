from memory.simple import SimpleMemory
import numpy as np

memory = SimpleMemory(100)


def create_random_trans_tensor():
    return memory.transform(np.random.rand(120, 160))


def test_fill_stack():
    rnd_tensor = create_random_trans_tensor()
    print(rnd_tensor.shape)
