import argparse
import six

def print_arguments(args, log):
    log.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        log.info('%s: %s' % (arg, value))
    log.info('------------------------------------------------')

def get_parser():

    parser = argparse.ArgumentParser(description='Hyperparameters')

    parser.add_argument('--PAD', default=0, type=int,
                        help='Index of PAD')
    parser.add_argument('--UNK', default=1, type=int,
                        help='Index of UNK')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size of train and evaluate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate of model layers')
    parser.add_argument('--layers')