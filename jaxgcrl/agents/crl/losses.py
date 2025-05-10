import flax.linen as nn
import jax
import jax.numpy as jnp

def mrn_distance(x, y, K=16):
    x_split = jnp.array_split(x, K, axis=-1)
    y_split = jnp.array_split(y, K, axis=-1)
    dists = [mrn_distance_component(x_split[i], y_split[i]) for i in range(K)]
    return jnp.stack(dists, axis=-1).mean(axis=-1)

def mrn_distance_component(x, y):
    eps = 1e-6
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = jnp.max(jax.nn.relu(x_prefix - y_prefix), axis=-1)
    l2_component = jnp.sqrt(jnp.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    assert max_component.shape == l2_component.shape
    return max_component + l2_component


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
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
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)  # dimension = B

        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )
        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)

        if config["energy_fn"] == "mrn":
            future_action = transitions.extras["future_action"]

            ga_repr = networks["sa_encoder"].apply(
                sa_encoder_params,
                jnp.concatenate([future_state, future_action], axis=-1),
            )
            dist = mrn_distance(sa_repr, ga_repr)
            qf_pi = -dist
        else:
            qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

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
        if config["energy_fn"] == "mrn":
            future_state = transitions.extras["future_state"]
            future_action = transitions.extras["future_action"]
            ga_repr = networks["sa_encoder"].apply(
                sa_encoder_params,  jnp.concatenate([future_state, future_action], axis=-1),
            )
            dist = mrn_distance(sa_repr[:, None], ga_repr[None])
            logits = -dist
        else:
            logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])

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
