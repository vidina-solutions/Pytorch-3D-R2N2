import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

trainable_params = []

def get_trainable_params():
    global trainable_params
    return trainable_params

class Weight(object):
    def __init__(self, w_shape, is_bias, mean=0, std=0.01, filler='msra',
                 fan_in=None, fan_out=None, name=None):
        super(Weight, self).__init__()
        assert (is_bias in [True, False])
        rng = np.random.RandomState()

        if isinstance(w_shape, collections.Iterable) and not is_bias:
            if len(w_shape) > 1 and len(w_shape) < 5:
                fan_in = np.prod(w_shape[1:])
                fan_out = np.prod(w_shape) / w_shape[1]
                n = (fan_in + fan_out) / 2.
            elif len(w_shape) == 5:
                # 3D Convolution filter
                fan_in = np.prod(w_shape[1:])
                fan_out = np.prod(w_shape) / w_shape[2]
                n = (fan_in + fan_out) / 2.
            else:
                raise NotImplementedError(
                    'Filter shape with ndim > 5 not supported: len(w_shape) = %d' % len(w_shape))
        else:
            n = 1

        if fan_in and fan_out:
            n = (fan_in + fan_out) / 2.

        if filler == 'gaussian':
            self.np_values = np.asarray(rng.normal(mean, std, w_shape), dtype=np.float32)
        elif filler == 'msra':
            self.np_values = np.asarray(
                rng.normal(mean, np.sqrt(2. / n), w_shape), dtype=np.float32)
        elif filler == 'xavier':
            scale = np.sqrt(3. / n)
            self.np_values = np.asarray(
                rng.uniform(
                    low=-scale, high=scale, size=w_shape), dtype=np.float32)
        elif filler == 'constant':
            self.np_values = np.cast[np.float32](mean * np.ones(
                w_shape, dtype=np.float32))
        elif filler == 'orth':
            ndim = np.prod(w_shape)
            W = np.random.randn(ndim, ndim)
            u, s, v = np.linalg.svd(W)
            self.np_values = u.astype(np.float32).reshape(w_shape)
        else:
            raise NotImplementedError('Filler %s not implemented' % filler)

        self.is_bias = is_bias  # Either the weight is bias or not
        self.val = nn.Parameter(torch.from_numpy(self.np_values))
        self.shape = w_shape
        self.name = name

        global trainable_params
        trainable_params.append(self)

class InputLayer(object):
    def __init__(self, input_shape, tinput=None):
        self._output_shape = input_shape
        self._input = tinput

    @property
    def output(self):
        if self._input is None:
            raise ValueError('Cannot call output for the layer. Initialize' \
                             + ' the layer with an input argument')
        return self._input

    @property
    def output_shape(self):
        return self._output_shape

class Layer(nn.Module):
    def __init__(self, prev_layer):
        super(Layer, self).__init__()
        self._output = None
        self._output_shape = None
        self._prev_layer = prev_layer
        self._input_shape = prev_layer.output_shape

    def set_output(self):
        raise NotImplementedError('Layer virtual class')

    @property
    def output_shape(self):
        if self._output_shape is None:
            raise ValueError('Set output shape first')
        return self._output_shape

    @property
    def output(self):
        if self._output is None:
            self.set_output()
        return self._output

class TensorProductLayer(Layer):
    def __init__(self, prev_layer, n_out, params=None, bias=True):
        super(TensorProductLayer, self).__init__(prev_layer)
        self._bias = bias
        n_in = self._input_shape[-1]

        if params is None:
            self.W = Weight((n_in, n_out), is_bias=False)
            if bias:
                self.b = Weight((n_out,), is_bias=True, mean=0.1, filler='constant')
        else:
            self.W = params[0]
            if bias:
                self.b = params[1]

        # parameters of the model
        self.params = [self.W]
        if bias:
            self.params.append(self.b)

        self._output_shape = [self._input_shape[0]]
        self._output_shape.extend(self._input_shape[1:-1])
        self._output_shape.append(n_out)

    def set_output(self):
        self._output = torch.matmul(self._prev_layer.output, self.W.val)
        if self._bias:
            self._output += self.b.val

# Define other layers and their set_output functions similarly

class SoftmaxWithLoss3D(nn.Module):
    def __init__(self, input):
        super(SoftmaxWithLoss3D, self).__init__()
        self.input = input
        self.exp_x = torch.exp(self.input)
        self.sum_exp_x = torch.sum(self.exp_x, dim=2, keepdim=True)

    def prediction(self):
        return self.exp_x / self.sum_exp_x

    def error(self, y, threshold=0.5):
        return torch.mean(torch.eq(torch.ge(self.prediction(), threshold), y))

    def loss(self, y):
        return torch.mean(
            torch.sum(-y * self.input, dim=2, keepdim=True) + torch.log(self.sum_exp_x))

class ConcatLayer(Layer):
    def __init__(self, prev_layers, axis=1):
        assert (len(prev_layers) > 1)
        super(ConcatLayer, self).__init__(prev_layers[0])
        self._axis = axis
        self._prev_layers = prev_layers

        self._output_shape = self._input_shape.copy()
        for prev_layer in prev_layers[1:]:
            self._output_shape[axis] += prev_layer._output_shape[axis]

    def set_output(self):
        self._output = torch.cat([x.output for x in self._prev_layers], dim=self._axis)

class LeakyReLU(Layer):
    def __init__(self, prev_layer, leakiness=0.01):
        super(LeakyReLU, self).__init__(prev_layer)
        self._leakiness = leakiness
        self._output_shape = self._input_shape

    def set_output(self):
        self._input = self._prev_layer.output
        if self._leakiness:
            self._output = F.leaky_relu(self._input, negative_slope=self._leakiness)
        else:
            self._output = F.relu(self._input)

class SigmoidLayer(Layer):
    def __init__(self, prev_layer):
        super(SigmoidLayer, self).__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = torch.sigmoid(self._prev_layer.output)

class TanhLayer(Layer):
    def __init__(self, prev_layer):
        super(TanhLayer, self).__init__(prev_layer)

    def set_output(self):
        self._output = torch.tanh(self._prev_layer.output)

class ComplementLayer(Layer):
    def __init__(self, prev_layer):
        super(ComplementLayer, self).__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = torch.ones_like(self._prev_layer.output) - self._prev_layer.output

# Create an instance of InputLayer and specify its shape
input_layer = InputLayer((batch_size, input_channels, input_height, input_width))

# Create other layers and specify their parameters as needed

# Set the next layer of each layer
tensor_product_layer.set_prev_layer(input_layer)
softmax_with_loss_layer.set_prev_layer(tensor_product_layer)

# Create the model by passing the last layer to nn.Module
model = nn.ModuleList([input_layer, tensor_product_layer, softmax_with_loss_layer])

# Use the model for forward pass
output = model[-1].output  # Get the output of the last layer
loss = model[-1].loss(y)  # Compute the loss

# Backpropagation and update of parameters
optimizer.zero_grad()
loss.backward()
optimizer.step()

