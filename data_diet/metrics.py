from flax import linen as nn
from jax import numpy as jnp


def logistic_loss(logits, labels):
  def logistic(x):
    return 1 / (1 + jnp.exp(-x))
  logits = jnp.clip(logistic(logits), 1e-12, 1-1e-12)
  return -jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1 -  logits))


def cross_entropy_loss(logits, labels):
  return jnp.mean(-jnp.sum(nn.log_softmax(logits) * labels, axis=-1))


def cross_entropy_loss_per_element(logits, labels):
  return -jnp.sum(nn.log_softmax(logits) * labels, axis=-1)


def correct(logits, labels):
  return jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)


def binary_correct(logits, labels):
  return jnp.where(logits > 0, 1, 0) == labels


def accuracy(logits, labels):
  return jnp.mean(correct(logits, labels))
