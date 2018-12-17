from comet_ml import Experiment
from utils.config_options import load_config
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Doom Using DQN")
    parser.add_argument("-k", "--key", required=True, help='comet api key')
    parser.add_argument("-c", "--config", required=True, help='path to config file')
    args = vars(parser.parse_args())

    config_options = load_config(args['config'])

    experiment = Experiment(api_key=args['key'], project_name=config_options['name'],
                            disabled=config_options['track_with_comet'])
    experiment.log_parameters(config_options)
