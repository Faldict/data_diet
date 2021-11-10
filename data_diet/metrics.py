from flax import linen as nn
from jax import numpy as jnp


def logistic_loss(logits, labels):
  logits = logits.squeeze()
  return jnp.mean(jnp.maximum(1 - 2. * (labels.astype(jnp.float32) -0.5) * logits, 0))


def constraints(logits, attributes):
  """ E[XZ] - E[X]E[Z]
  """
  return jnp.mean(logits * attributes) - jnp.mean(logits) * jnp.mean(attributes)


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


def fairness(logits, labels, attributes):
  return jnp.sum(logits[attributes > 0]), jnp.sum(logits[attributes == 0])