from cs231n.layers import *
from cs231n.fast_layers import *

def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_bn_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param, use_bn, use_dropout):
  bn_cache = None
  dropout_cache = None

  fc, fc_cache = affine_forward(x, w, b)
  if use_bn:
      fc, bn_cache = batchnorm_forward(fc, gamma, beta, bn_param)
  out, relu_cache = relu_forward(fc)
  if use_dropout:
      out, dropout_cache = dropout_forward(out, dropout_param)
  cache = (fc_cache, bn_cache, relu_cache, dropout_cache)
  return out, cache


def affine_bn_relu_dropout_backward(dout, cache, use_bn, use_dropout):
  fc_cache, bn_cache, relu_cache, dropout_cache = cache
  dgamma = None
  dbeta = None
  dx = dout.copy()
  if use_dropout:
      dx = dropout_backward(dx, dropout_cache)
  dx = relu_backward(dx, relu_cache)
  if use_bn:
      dx, dgamma, dbeta = batchnorm_backward_alt(dx, bn_cache)
  dx, dw, db = affine_backward(dx, fc_cache)
  return dx, dw, db, dgamma, dbeta


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
  """
  Convenience layer that performs a convolution, a BN, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer
  - gamma, beta, bn_param: Weights and parameters for the bn layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  conv_cache, bn_cache, relu_cache, pool_cache = None, None, None, None

  out, conv_cache = conv_forward_fast(x, w, b, conv_param)
  if bn_param is not None:
      out, bn_cache = spatial_batchnorm_forward(out, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out)
  if pool_param is not None:
      out, pool_cache = max_pool_forward_fast(out, pool_param)

  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache


def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  dgamma, dbeta = None, None

  dtmp = dout.copy()
  if pool_cache is not None:
      dtmp = max_pool_backward_fast(dtmp, pool_cache)
  dtmp = relu_backward(dtmp, relu_cache)
  if bn_cache is not None:
      dtmp, dgamma, dbeta = spatial_batchnorm_backward(dtmp, bn_cache)
  dx, dw, db = conv_backward_fast(dtmp, conv_cache)

  return dx, dw, db, dgamma, dbeta
