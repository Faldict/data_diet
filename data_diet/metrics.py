from flax import linen as nn
from jax import numpy as jnp


def logistic_loss(logits, labels):
  logits = logits.squeeze()
  # logits = nn.sigmoid(logits)
  # return -jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1 -  logits))
  return jnp.mean(jnp.maximum(1 - 20. * (labels.astype(jnp.float32) -0.5) * logits, 0))
  # return jnp.mean(jnp.exp((labels - 0.5) * logits))

def cross_entropy_loss(logits, labels):
  return jnp.mean(-jnp.sum(nn.log_softmax(logits) * labels, axis=-1))


def cross_entropy_loss_per_element(logits, labels):
  return -jnp.sum(nn.log_softmax(logits) * labels, axis=-1)


def correct(logits, labels):
  return jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)


def binary_correct(logits, labels):
  return (logits.squeeze() > 0) == labels


def accuracy(logits, labels):
  return jnp.mean(correct(logits, labels))
