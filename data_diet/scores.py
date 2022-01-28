from .data import get_class_balanced_random_subset
from .gradients import compute_mean_logit_gradients, flatten_jacobian, get_mean_logit_gradients_fn
from .metrics import constraints, cross_entropy_loss, logistic_loss
import flax.linen as nn
from jax import jacrev, jit, vmap
import jax.numpy as jnp
import numpy as np


def get_lord_error_fn(fn, params, state, ord):
  @jit
  def lord_error(X, Y):
    errors = nn.softmax(fn(params, state, X)) - Y
    scores = jnp.linalg.norm(errors, ord=ord, axis=-1)
    return scores
  np_lord_error = lambda X, Y: np.array(lord_error(X, Y))
  return np_lord_error


def get_margin_error(fn, params, state, score_type):
  fn_jit = jit(lambda X: fn(params, state, X))

  def margin_error(X, Y):
    batch_sz = X.shape[0]
    P = np.array(nn.softmax(fn_jit(X)))
    correct_logits = Y.astype(bool)
    margins = P[~correct_logits].reshape(batch_sz, -1) - P[correct_logits].reshape(batch_sz, 1)
    if score_type == 'max':
      scores = np.max(margins, -1)
    elif score_type == 'sum':
      scores = np.sum(margins, -1)
    return scores

  return margin_error


def get_grad_norm_fn(fn, params, state):

  @jit
  def score_fn(X, Y):
    per_sample_loss_fn = lambda p, x, y: vmap(cross_entropy_loss)(fn(p, state, x), y)
    loss_grads = flatten_jacobian(jacrev(per_sample_loss_fn)(params, X, Y))
    scores = jnp.linalg.norm(loss_grads, axis=-1)
    return scores

  return lambda X, Y: np.array(score_fn(X, Y))



def get_hinge_loss_grad_norm_fn(fn, params, state):

  @jit
  def score_fn(X, Y):
    per_sample_loss_fn = lambda p, x, y: vmap(logistic_loss)(fn(p, state, x), y)
    loss_grads = flatten_jacobian(jacrev(per_sample_loss_fn)(params, X, Y))
    scores = jnp.linalg.norm(loss_grads, axis=-1)
    return scores

  return lambda X, Y: np.array(score_fn(X, Y))


def get_covariance_fn(fn, params, state):

  @jit
  def score_fn(X, Y, Z):
    # per_sample_loss_fn = lambda p, x, y, z: vmap(constraints)(fn(p, state, x), z)
    per_sample_fn = lambda p, x, y, z: vmap(lambda logits, attributes: logits * (attributes - Z.mean()))(fn(p, state, x), z)
    grads = flatten_jacobian(jacrev(per_sample_fn)(params, X, Y, Z))
    kernel = grads @ grads.T
    score = jnp.mean(kernel, axis=-1)
    return score

  return lambda X, Y, Z: np.array(score_fn(X, Y, Z))


def get_score_fn(fn, params, state, score_type):
  if score_type == 'l2_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, params, state, 2)
  elif score_type == 'l1_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, params, state, 1)
  elif score_type == 'max_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, params, state, 'max')
  elif score_type == 'sum_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, params, state, 'sum')
  elif score_type == 'grad_norm':
    print(f'compute {score_type}...')
    score_fn = get_grad_norm_fn(fn, params, state)
  else:
    raise NotImplementedError
  return score_fn


def get_fair_score_fn(fn, params, state, score_type):

  if score_type == 'covariance':
    print(f'compute {score_type}...')
    score_fn = get_covariance_fn(fn, params, state)
  elif score_type == 'grad_norm':
    print(f'compute {score_type}...')
    score_fn = lambda x, y, z: get_hinge_loss_grad_norm_fn(fn, params, state)(x, y)
  elif score_type == 'l2_error':
    print(f'compute {score_type}...')
    score_fn = lambda x, y, z: get_lord_error_fn(fn, params, state, 2)(x, y)
  elif score_type == 'l1_error':
    print(f'compute {score_type}...')
    score_fn = lambda x, y, z: get_lord_error_fn(fn, params, state, 1)(x, y)
  else:
    raise NotImplementedError
  return score_fn


def compute_scores(fn, params, state, X, Y, batch_sz, score_type):
  n_batches = X.shape[0] // batch_sz
  Xs, Ys = np.array_split(X, n_batches), np.array_split(Y, n_batches)
  score_fn = get_score_fn(fn, params, state, score_type)
  scores = []
  for i, (X, Y) in enumerate(zip(Xs, Ys)):
    print(f'score batch {i+1} of {n_batches}')
    scores.append(score_fn(X, Y))
  scores = np.concatenate(scores)
  return scores


def compute_fair_scores(fn, params, state, X, Y, Z, batch_size, score_type):
  n_batches = X.shape[0] // batch_size
  Xs, Ys, Zs = np.array_split(X, n_batches), np.array_split(Y, n_batches), np.array_split(Z, n_batches)
  score_fn = get_fair_score_fn(fn, params, state, score_type)
  scores = []
  for i, (x, y, z) in enumerate(zip(Xs, Ys, Zs)):
    print(f'score batch {i+1} of {n_batches}, group balance: {z.mean()*100:.2f}%')
    scores.append(score_fn(x, y, z))
  scores = np.concatenate(scores)
  return scores


def compute_unclog_scores(fn, params, state, X, Y, cls_smpl_sz, seed, batch_sz_mlgs):
  n_batches = X.shape[0]
  Xs = np.split(X, n_batches)
  X_mlgs, _ = get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed)
  mlgs = compute_mean_logit_gradients(fn, params, state, X_mlgs, batch_sz_mlgs)
  logit_grads_fn = get_mean_logit_gradients_fn(fn, params, state)
  score_fn = jit(lambda X: jnp.linalg.norm((logit_grads_fn(X) - mlgs).sum(0)))
  scores = []
  for i, X in enumerate(Xs):
    if i % 500 == 0: print(f'images {i} of {n_batches}')
    scores.append(score_fn(X).item())
  scores = np.array(scores)
  return scores
