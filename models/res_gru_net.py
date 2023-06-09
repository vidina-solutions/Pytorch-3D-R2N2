import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualGRUNet(nn.Module):
    def __init__(self):
        super(ResidualGRUNet, self).__init__()
        
        n_gru_vox = 4
        n_convfilter = [96, 128, 256, 256, 256, 256]
        n_fc_filters = [1024]
        n_deconvfilter = [128, 128, 128, 64, 32, 2]
        
        self.conv1a = nn.Conv2d(3, n_convfilter[0], kernel_size=(7, 7))
        self.conv1b = nn.Conv2d(n_convfilter[0], n_convfilter[0], kernel_size=(3, 3))

        self.conv2a = nn.Conv2d(n_convfilter[0], n_convfilter[1], kernel_size=(3, 3))
        self.conv2b = nn.Conv2d(n_convfilter[1], n_convfilter[1], kernel_size=(3, 3))

        # ...
        # Add the remaining layers following the same pattern, up to pool6 and fc7
        # ...

        # Define GRU in 3D and convolutional layers in 3D as well
        self.gru = nn.GRU(n_gru_vox, n_deconvfilter[0])
        self.conv3d = nn.Conv3d(n_deconvfilter[0], n_deconvfilter[1], kernel_size=(3, 3, 3))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, 2)

        # ...
        # Pass x through the remaining layers
        # ...

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc(x))

        # Reshape x for GRU, and pass it through GRU and subsequent layers
        x = x.view(x.size(0), -1, self.n_gru_vox)
        x, _ = self.gru(x)
        x = F.relu(self.conv3d(x))
        x = self.up(x)

        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

