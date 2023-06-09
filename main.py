#!/usr/bin/env python
import sys
import argparse
import pprint
import logging
import multiprocessing as mp
import torch

from lib.config import cfg, cfg_from_file, cfg_from_list
from lib.test_net import test_net
from lib.train_net import train_net


def parse_args():
    parser = argparse.ArgumentParser(description='Main 3Deverything train/test file')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [gpu0]', default=cfg.CONST.DEVICE, type=str)
    # ...
    # Other arguments are the same
    # ...

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    # Set main GPU device
    torch.cuda.set_device(args.gpu_id)

    # ...
    # Configuration management code remains the same
    # ...

    print('Using config:')
    pprint.pprint(cfg)

    if not args.test:
        train_net()
    else:
        test_net()


if __name__ == '__main__':
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    main()

