import numpy as np
import datetime as dt
import torch
import torch.nn as nn

from lib.config import cfg


class Net(nn.Module):

    def __init__(self, random_seed=dt.datetime.now().microsecond):
        super(Net, self).__init__()
        self.rng = np.random.RandomState(random_seed)

        self.batch_size = cfg.CONST.BATCH_SIZE
        self.img_w = cfg.CONST.IMG_W
        self.img_h = cfg.CONST.IMG_H
        self.n_vox = cfg.CONST.N_VOX

        # self.x and self.y will be created dynamically during forward pass

        self.activations = []  # list of all intermediate activations
        self.loss = None  # final loss
        self.output = None  # final output
        self.error = None  # final output error

        self.setup()

    def setup(self):
        self.network_definition()

    def network_definition(self):
        """ A child network must define
        self.loss
        self.error
        self.output
        self.activations is optional
        """
        raise NotImplementedError("Virtual Function")

    def add_layer(self, layer):
        raise NotImplementedError("TODO: add a layer")

    def forward(self, x, y):
        """ A child network must implement its forward function"""
        raise NotImplementedError("Virtual Function")

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print('saving network parameters to ' + filename)

    def load(self, filename):
        print('loading network parameters from ' + filename)
        try:
            self.load_state_dict(torch.load(filename))
        except Exception as e:
            print(f"Error while loading model: {e}")

