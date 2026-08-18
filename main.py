import argparse
import importlib
import yaml

TRAIN_MAP = {
    ('cyclegan', 'default'):    'models.CycleGAN.train',
    ('dscms',    'default'):    'models.DSCMS.train',
    ('prusr',    'default'):    'models.PRUSR.train',
    ('srcnn',    'default'):    'models.SRCNN.train',
    ('srno',    'default'):    'models.srno.train',
    ('sofono',    'default'):    'models.sofono.train2',
    ('cyclegan', 'multistage'): 'models.CycleGAN.train',  # update when available
    ('dscms',    'multistage'): 'models.DSCMS.train',     # update when available
    ('prusr',    'multistage'): 'models.PRUSR.train',     # update when available
    ('srcnn',    'multistage'): 'models.SRCNN.msrnn_train',
    ('srno',    'multistage'):    'models.srno.train',       # update when available
    ('sofono',    'multistage'):    'models.sofono.train',  # update when available
}

def main():
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--model', choices=['cyclegan', 'dscms', 'prusr', 'srcnn', 'srno', 'sofono'],
                        help='Model to train')
    parser.add_argument('--training-strategy', choices=['default', 'multistage'],
                        default='default', help='Training strategy to use')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        experiment_config = yaml.safe_load(file)

    strategy = args.training_strategy
    module_path = TRAIN_MAP[(args.model, strategy)]


    train = importlib.import_module(module_path)
    train.main(experiment_config)

if __name__ == "__main__":
    main()