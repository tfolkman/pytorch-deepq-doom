## TRAINING

1. Make sure have [pipenv](https://pipenv.readthedocs.io/en/latest/) installed.
2. Run: 'pipenv install' to install the required libraries into a virtualenv. Note: you may need to install additional
non python dependencies, especially for [Vizdoom](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) 
and [Pytorch](https://pytorch.org/)
3. Sign up for [comet ml](https://www.comet.ml/) (free). A very nice service for tracking your experiments. This repo
automatically uses comet to track the loss and reward during training. Grab your API key.
4. Modify the config.json as you see fit:
    * disable_comet: Whether to turn off comet tracking (note still need to pass an API key value)
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
    * save_every: how often to save the model weights and log statistics to the command line (# episodes)
    * update_target_every: how often to update the target model (number of games).
    * save_file: where to save model weights to
5. Run: 'pipenv run python --key {comet api key} --config {path to config.json}'
    * Note: path to config.json defaults to config.json
6. Go check out your comet dashboard and track its progress!


## Extendability

I have tried to make this package fairly easy to extend by creating abstract base classes for the game, memory, 
and loss. If you want to change any of these, creating classes that implement the abstract base class should work.
Then you just have to update the train.py file to use your implementation.

To implement a different model, just implement nn.module as is standard for pytorch.

## Running Trained Model

Run 'pipenv run python --weights {path to trained weights} --n {number of games to play} --sleep {how long (seconds) to 
sleep between moves} run.py'

## Using my pre-trained weights

You can download some weights I trained [here](https://www.dropbox.com/s/itza42i81toutfc/doom_dqn_stacked.state?dl=0).

Download them; create a folder called "model_weights" and move them into that folder. Then you can run 'pipenv run python run.py' 

You can view the training for this model, as well as the hyper-parameters, on comet ml [here](https://www.comet.ml/syrios/doom-deepq/47f0975ad66e49d7b61f117912bf9d7e).

## Sources
* https://simoninithomas.github.io/Deep_reinforcement_learning_Course/


  
