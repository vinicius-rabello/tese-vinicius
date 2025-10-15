import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('model', choices=['cyclegan', 'dscms', 'prusr'],
                        help='Model to train')
    args = parser.parse_args()

    if args.model == 'cyclegan':
        from models.CycleGAN import train
        train.main()
    elif args.model == 'dscms':
        from models.DSCMS import train
        train.main()
    elif args.model == 'prusr':
        from models.PRUSR import train
        train.main()

if __name__ == "__main__":
    main()