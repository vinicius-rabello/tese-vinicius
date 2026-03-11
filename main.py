import argparse

TRAIN_MAP = {
    ('cyclegan', 'default'):    'models.CycleGAN.train',
    ('dscms',    'default'):    'models.DSCMS.train',
    ('prusr',    'default'):    'models.PRUSR.train',
    ('srcnn',    'default'):    'models.SRCNN.train',
    ('cyclegan', 'multistage'): 'models.CycleGAN.train',  # update when available
    ('dscms',    'multistage'): 'models.DSCMS.train',     # update when available
    ('prusr',    'multistage'): 'models.PRUSR.train',     # update when available
    ('srcnn',    'multistage'): 'models.SRCNN.msrnn_train',
}

def main():
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--model', choices=['cyclegan', 'dscms', 'prusr', 'srcnn'],
                        help='Model to train')
    parser.add_argument('--training-strategy', choices=['default', 'multistage'],
                        default='default', help='Training strategy to use')
    args = parser.parse_args()

    strategy = args.training_strategy
    module_path = TRAIN_MAP[(args.model, strategy)]

    import importlib
    train = importlib.import_module(module_path)
    train.main()

if __name__ == "__main__":
    main()