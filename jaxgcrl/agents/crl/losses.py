import flax.linen as nn
import jax
import jax.numpy as jnp

MAX_LOGIT_SCALE = 100.0  # matches CLIP's clamp on the learned temperature


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        # NOTE: norm is taken over the last (feature) axis so this is a per-pair
        # cosine similarity, not a single global scalar over the whole batch.
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def apply_logit_scale(config, critic_params, energy):
    """Cosine similarity is bounded to [-1, 1], which is too small a logit range
    for InfoNCE to produce a useful (non-near-uniform) softmax. Rescale it by a
    learned temperature, following CLIP; other energy functions are unbounded
    and left as-is.
    """
    if config["energy_fn"] != "cosine":
        return energy
    scale = jnp.exp(jnp.minimum(critic_params["log_logit_scale"], jnp.log(MAX_LOGIT_SCALE)))
    return energy * scale


def _matmul_t(x, y):
    """x @ y.T at full float32 precision. Plain `@`/`jnp.dot` let XLA pick a
    reduced-precision (e.g. TF32-style) algorithm on GPU matmul hardware for
    speed, which is fine for most matmuls but introduces up to ~0.4% relative
    error here -- too much for a training signal we want left unchanged.
    `HIGHEST` recovers float32-accumulation-order-level agreement (~1e-5)
    with the elementwise reduction this replaces, at a modest (not zero)
    speed cost.
    """
    return jnp.matmul(x, y.T, precision=jax.lax.Precision.HIGHEST)


def pairwise_energy_fn(name, x, y):
    """All-pairs (B, B) energy matrix for x, y of shape (B, D), equivalent to
    `energy_fn(name, x[:, None, :], y[None, :, :])` but computed from B*D-sized
    matmuls/norms instead of ever materializing the broadcasted (B, B, D)
    intermediate that the naive elementwise-multiply-then-sum requires. That
    intermediate is what makes computing all-pairs energies for a batch of
    size B expensive (both in memory and compute) as repr_dim D grows, since
    a GEMM never needs it.
    """
    if name == "dot":
        return _matmul_t(x, y)
    elif name == "cosine":
        # per-row norms outer-producted against each other, matching
        # energy_fn("cosine", x[:, None, :], y[None, :, :])'s axis=-1 norms
        x_norm = jnp.linalg.norm(x, axis=-1)[:, None]
        y_norm = jnp.linalg.norm(y, axis=-1)[None, :]
        return _matmul_t(x, y) / (x_norm * y_norm + 1e-6)
    elif name == "l2":
        # ||x_i - y_j||^2 = ||x_i||^2 - 2 * x_i . y_j + ||y_j||^2
        x_sq = jnp.sum(x**2, axis=-1)[:, None]
        y_sq = jnp.sum(y**2, axis=-1)[None, :]
        # clip: floating-point cancellation in the expansion above can
        # otherwise push near-zero distances slightly negative
        return -jnp.maximum(x_sq - 2 * _matmul_t(x, y) + y_sq, 0.0)
    elif name == "norm":
        # Deliberately NOT reformulated via the ||x-y||^2 = ||x||^2 - 2xy +
        # ||y||^2 expansion used for "l2" above. That expansion mixes a
        # matmul-computed cross term with elementwise-computed squared norms,
        # which use different float32 summation orders; the resulting ~1e-5
        # discrepancy is negligible for "l2" but gets blown up by this
        # branch's sqrt (unbounded derivative near 0) into a >5x relative
        # error exactly on near-identical x/y -- i.e. exactly on correctly
        # matched positive pairs, which is the one place this metric most
        # needs to be accurate. Falls back to the O(batch^2 * repr_dim)
        # broadcast path.
        return energy_fn(name, x[:, None, :], y[None, :, :])
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        obs = transitions.observation  # expected_shape = self.batch_size, obs_size + goal_size
        state = obs[:, : config["state_size"]]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)  # dimension = B

        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )
        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)

        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)
        qf_pi = apply_logit_scale(config, critic_params, qf_pi)

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }

    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
    def critic_loss(critic_params, transitions, key):
        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )

        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, transitions.observation[:, config["state_size"] :]
        )

        # InfoNCE
        logits = pairwise_energy_fn(config["energy_fn"], sa_repr, g_repr)
        logits = apply_logit_scale(config, critic_params, logits)
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }

    return training_state, metrics
