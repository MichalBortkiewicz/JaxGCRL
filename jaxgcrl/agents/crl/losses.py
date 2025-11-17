import flax.linen as nn
import jax
import jax.numpy as jnp


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

        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

        per_sample_loss = jnp.exp(log_alpha) * log_prob - qf_pi

        actor_loss = jnp.mean(per_sample_loss)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)
    
    batch_size = transitions.observation.shape[0]
    key, subkey = jax.random.split(key)
    sample_keys = jax.random.split(subkey, batch_size)

    (batch_actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )

    def single_sample_grad(i, key):
        single_transition = jax.tree_util.tree_map(lambda x: x[i], transitions)
        single_transition = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), single_transition)

        grad_fn = jax.grad(actor_loss, has_aux=True)
        return grad_fn(
            training_state.actor_state.params,
            training_state.critic_state.params,
            training_state.alpha_state.params["log_alpha"],
            single_transition,
            key
        )
    per_sample_grads, _ = jax.vmap(single_sample_grad)(jnp.arange(batch_size), sample_keys)

    was_proposed_mask = transitions.extras["state_extras"]["was_proposed_goal_mask"]

    def flatten_single_grad(i):
        single_grad = jax.tree_map(lambda x: x[i], per_sample_grads)
        flat, _ = jax.flatten_util.ravel_pytree(single_grad)
        return flat
    
    all_grads_flat = jax.vmap(flatten_single_grad)(jnp.arange(batch_size))  # (batch_size, total_params)
    
    num_env = jnp.sum(1 - was_proposed_mask)
    num_rb = jnp.sum(was_proposed_mask)
    env_mask = (1 - was_proposed_mask)[:, None]  # (batch_size, 1)
    proposed_mask = was_proposed_mask[:, None]

    d_env_grad_mean = jnp.sum(all_grads_flat * env_mask, axis=0) / (num_env + 1e-8)
    d_rb_grad_mean = jnp.sum(all_grads_flat * proposed_mask, axis=0) / (num_rb + 1e-8)

    env_grads_centered = all_grads_flat - d_env_grad_mean[None, :]
    d_env_grad_var = jnp.sum((env_grads_centered**2 * env_mask).sum(axis=0)) / (num_env + 1e-8)  # tr Var(d_env)
    rb_grads_centered = all_grads_flat - d_rb_grad_mean[None, :]
    d_rb_grad_var = jnp.sum((rb_grads_centered**2 * proposed_mask).sum(axis=0)) / (num_rb + 1e-8)  # tr Var(d_rb)

    # Update actor
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    batch_alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": batch_actor_loss,
        "alpha_loss": batch_alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
        "rb_grad_trvar": d_rb_grad_var,
        "env_grad_trvar": d_env_grad_var,
        "rb_grad_mean_norm": jnp.linalg.norm(d_rb_grad_mean),
        "env_grad_mean_norm": jnp.linalg.norm(d_env_grad_mean),
        "env_rb_bias_squared": jnp.sum((d_env_grad_mean - d_rb_grad_mean) ** 2),
        "num_rb_samples": num_rb,
        "num_env_samples": num_env,
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
