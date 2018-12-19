## TRAINING

1. Make sure have [pipenv](https://pipenv.readthedocs.io/en/latest/) installed.
2. Run: 'pipenv install' to install the required libraries into a virtualenv. Note: you may need to install additional
non python dependencies, especially for [Vizdoom](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) 
and [Pytorch](https://pytorch.org/)
3. Sign up for [comet ml](https://www.comet.ml/) (free). A very nice service for tracking your experiments. This repo
automatically uses comet to track the loss and reward during training. Grab your API key.
4. Modify the config.json as you see fit:
    * track_with_comet: Whether to turn on comet tracking (note still need to pass an API key value)
    * name: project name in comet
    * lr: learning rate (I assume adam optimizer)
    * total_episodes: the number of games to play during training
    * max_steps: max moves per a game
    * batch_size: batch size
    * explore_start: the probability of exploration at the start
    * explore_end: the lowest value for probability of exploration
    * decay: how many steps to decay the learning rate over
    * gamma: the discount factor for future rewards
    * memory_size: max memory size
    * save_every: how often to save the model weights
    * save_path: where to save model weights to
5. Run: 'pipenv python -k {comet api key} -c {path to config.json}'
    * Note: path to config.json defaults to config.json
6. Go check out your comet dashboard and track its progress!


## Extendability

## Running Trained Model

## Sources
* https://simoninithomas.github.io/Deep_reinforcement_learning_Course/


  
