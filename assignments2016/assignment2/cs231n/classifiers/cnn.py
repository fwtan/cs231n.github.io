import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = np.random.normal(0.0, weight_scale,
        (num_filters, input_dim[0], filter_size, filter_size))
    self.params['b1'] = np.zeros((num_filters,))

    pad = (filter_size - 1) / 2
    stride = 1

    nH = 1 + (input_dim[1] + 2 * pad - filter_size)/stride
    nW = 1 + (input_dim[2] + 2 * pad - filter_size)/stride

    nH /= 2
    nW /= 2

    tmp_dims = nH * nW * num_filters

    self.params['W2'] = np.random.normal(0.0, weight_scale, (tmp_dims, hidden_dim))
    self.params['b2'] = np.zeros((hidden_dim,))

    self.params['W3'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros((num_classes,))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
        self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    fc1_out, fc1_cache   = affine_relu_forward(conv_out, W2, b2)
    scores, scores_cache = affine_forward(fc1_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
        return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

    dscores, dW3, grads['b3'] = affine_backward(dout, scores_cache)
    grads['W3'] = dW3 + self.reg * W3
    dfc1, dW2, grads['b2'] = affine_relu_backward(dscores, fc1_cache)
    grads['W2'] = dW2 + self.reg * W2
    dX, dW1, grads['b1'] = conv_relu_pool_backward(dfc1, conv_cache)
    grads['W1'] = dW1 + self.reg * W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass

class MultiLayerConvNet(object):
    def __init__(self, conv_filter_dims, hidden_dim=100,
                 input_dim=(3, 32, 32), num_classes=10,
                 dropout=0.0,
                 use_batchnorm=True,
                 reg=0.0,
                 weight_scale=5e-3,
                 dtype=np.float32,
                 seed=None):

        # super(MultiLayerConvNet, self).__init__()

        self.num_conv_layers = len(conv_filter_dims)
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.use_max_poolings = [False] * self.num_conv_layers
        self.reg = reg
        self.dtype = dtype
        self.params = {}

        C, H, W = input_dim
        num_filters = [x[0] for x in conv_filter_dims]
        num_filters.insert(0, C)

        ########################################################################
        # dropout pool conv and bn params
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        self.conv_params = []
        self.conv_params = [{'stride': 1, 'pad': (x[1] - 1)/2} for x in conv_filter_dims]

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_conv_layers + 1)]
        ########################################################################

        ########################################################################
        # conv and bn weights
        for i in xrange(0, self.num_conv_layers):
            filter_size = conv_filter_dims[i][1]
            use_max_pooling = conv_filter_dims[i][2]
            pad = (filter_size - 1)/2
            stride = 1

            wi = 'W{:d}'.format(i)
            bi = 'b{:d}'.format(i)

            self.params[wi] = np.random.normal(0.0, weight_scale, (num_filters[i+1], num_filters[i], filter_size, filter_size))
            self.params[bi] = np.zeros((num_filters[i+1],))

            if self.use_batchnorm:
                gammai = 'gamma{:d}'.format(i)
                betai = 'beta{:d}'.format(i)
                self.params[gammai] = np.ones(num_filters[i+1])
                self.params[betai] = np.zeros(num_filters[i+1])

            H = 1 + (H + 2 * pad - filter_size)/stride
            W = 1 + (W + 2 * pad - filter_size)/stride

            if use_max_pooling:
                self.use_max_poolings[i] = True
                H /= 2
                W /= 2

        assert (H > 1 and W > 1)

        tmp_dims = H * W * num_filters[self.num_conv_layers]

        wh = 'W{:d}'.format(self.num_conv_layers)
        bh = 'b{:d}'.format(self.num_conv_layers)
        self.params[wh] = np.random.normal(0.0, weight_scale, (tmp_dims, hidden_dim))
        self.params[bh] = np.zeros((hidden_dim,))

        if self.use_batchnorm:
            gammai = 'gamma{:d}'.format(self.num_conv_layers)
            betai = 'beta{:d}'.format(self.num_conv_layers)
            self.params[gammai] = np.ones(hidden_dim)
            self.params[betai] = np.zeros(hidden_dim)

        ws = 'W{:d}'.format(self.num_conv_layers + 1)
        bs = 'b{:d}'.format(self.num_conv_layers + 1)
        self.params[ws] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params[bs] = np.zeros((num_classes,))
        ########################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        ########################################################################
        # Forward pass
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        caches = []
        x = X.copy()
        for i in xrange(0, self.num_conv_layers):
            gamma, beta, bn_param, pool_param = None, None, None, None
            w = self.params['W%d'%i]
            b = self.params['b%d'%i]
            if self.use_batchnorm:
                gamma = self.params['gamma%d'%i]
                beta = self.params['beta%d'%i]
                bn_param = self.bn_params[i]
            if self.use_max_poolings[i]:
                pool_param = self.pool_param
            x, c = conv_bn_relu_pool_forward(x, w, b, gamma, beta, self.conv_params[i], bn_param, pool_param)
            caches.append(c)

        # hidden layer
        wh = 'W{:d}'.format(self.num_conv_layers)
        bh = 'b{:d}'.format(self.num_conv_layers)
        w = self.params[wh]
        b = self.params[bh]
        gamma, beta, bn_param, dropout_param = None, None, None, None
        if self.use_batchnorm:
            gammah = 'gamma{:d}'.format(self.num_conv_layers)
            betah = 'beta{:d}'.format(self.num_conv_layers)
            gamma = self.params[gammah]
            beta = self.params[betah]
            bn_param = self.bn_params[self.num_conv_layers]
        if self.use_dropout:
            dropout_param = self.dropout_param

        x, c = affine_bn_relu_dropout_forward(x, w, b, gamma, beta,
                                            bn_param, dropout_param,
                                            self.use_batchnorm, self.use_dropout)
        caches.append(c)

        ws = 'W{:d}'.format(self.num_conv_layers + 1)
        bs = 'b{:d}'.format(self.num_conv_layers + 1)
        w = self.params[ws]
        b = self.params[bs]

        scores, c = affine_forward(x, w, b)
        caches.append(c)

        # If test mode return early
        if mode == 'test':
            return scores
        ########################################################################

        ########################################################################
        # Back propogation
        loss, grads = 0.0, {}

        # output and last layer
        loss, dout = softmax_loss(scores, y)
        dout, dw, db = affine_backward(dout, caches[-1])
        ws = 'W{:d}'.format(self.num_conv_layers + 1)
        bs = 'b{:d}'.format(self.num_conv_layers + 1)
        w = self.params[ws]
        grads[ws] = dw + self.reg * w
        grads[bs] = db
        loss += 0.5 * self.reg * np.sum(w * w)

        ws = 'W{:d}'.format(self.num_conv_layers)
        bs = 'b{:d}'.format(self.num_conv_layers)
        w = self.params[ws]
        dout, dw, db, dgamma, dbeta = affine_bn_relu_dropout_backward(dout, caches[-2], self.use_batchnorm, self.use_dropout)
        if self.use_batchnorm:
            gammah = 'gamma{:d}'.format(self.num_conv_layers)
            betah = 'beta{:d}'.format(self.num_conv_layers)
            grads[gammah] = dgamma
            grads[betah] = dbeta
        grads[ws] = dw + self.reg * w
        grads[bs] = db
        loss += 0.5 * self.reg * np.sum(w * w)

        for i in reversed(xrange(0, self.num_conv_layers)):
            dout, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dout, caches[i])
            ws = 'W{:d}'.format(i)
            bs = 'b{:d}'.format(i)
            w = self.params[ws]
            if self.use_batchnorm:
                gammah = 'gamma{:d}'.format(i)
                betah = 'beta{:d}'.format(i)
                grads[gammah] = dgamma
                grads[betah] = dbeta
            grads[ws] = dw + self.reg * w
            grads[bs] = db
            loss += 0.5 * self.reg * np.sum(w * w)

        return loss, grads
