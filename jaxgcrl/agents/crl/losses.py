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

        # Handle both single critic and ensemble cases
        is_ensemble = isinstance(critic_params["sa_encoder"], list)
        
        if is_ensemble:
            # For ensemble, compute mean Q-value across all critics
            all_qf_pi = []
            for i in range(len(critic_params["sa_encoder"])):
                sa_encoder_params = critic_params["sa_encoder"][i]
                g_encoder_params = critic_params["g_encoder"][i]
                sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
                g_repr = networks["g_encoder"].apply(g_encoder_params, goal)
                qf_pi_i = energy_fn(config["energy_fn"], sa_repr, g_repr)
                all_qf_pi.append(qf_pi_i)
            qf_pi = jnp.mean(jnp.stack(all_qf_pi, axis=0), axis=0)
        else:
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
    }

    # Only compute per-sample gradient statistics if adaptive mixing is enabled
    if config.get("use_adaptive_mixing", False):
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

        metrics.update({
            "adaptive_mixing/rb_grad_trvar": d_rb_grad_var,
            "adaptive_mixing/env_grad_trvar": d_env_grad_var,
            "adaptive_mixing/rb_grad_mean_norm": jnp.linalg.norm(d_rb_grad_mean),
            "adaptive_mixing/env_grad_mean_norm": jnp.linalg.norm(d_env_grad_mean),
            "adaptive_mixing/env_rb_bias_squared": jnp.sum((d_env_grad_mean - d_rb_grad_mean) ** 2),
            "adaptive_mixing/num_rb_samples": num_rb,
            "adaptive_mixing/num_env_samples": num_env,
        })

    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
    """Update critic(s). Supports both single critic and ensemble of critics."""
    
    critic_params = training_state.critic_state.params
    is_ensemble = isinstance(critic_params["sa_encoder"], list)
    
    def single_critic_loss(sa_params, g_params, transitions, key):
        """Loss for a single critic."""
        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(sa_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(
            g_params, transitions.observation[:, config["state_size"] :]
        )

        # InfoNCE
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return loss, (logsumexp, correct, logits_pos, logits_neg)
    
    if is_ensemble:
        # Ensemble case: train each critic independently with the full optimizer
        num_ensemble = len(critic_params["sa_encoder"])
        
        def ensemble_loss(critic_params, transitions, key):
            """Combined loss for all ensemble members."""
            total_loss = 0.0
            all_logsumexp = []
            all_correct = []
            all_logits_pos = []
            all_logits_neg = []
            
            for i in range(num_ensemble):
                sa_params = critic_params["sa_encoder"][i]
                g_params = critic_params["g_encoder"][i]
                loss, (logsumexp, correct, logits_pos, logits_neg) = single_critic_loss(
                    sa_params, g_params, transitions, key
                )
                total_loss += loss
                all_logsumexp.append(logsumexp)
                all_correct.append(correct)
                all_logits_pos.append(logits_pos)
                all_logits_neg.append(logits_neg)
            
            avg_loss = total_loss / num_ensemble
            avg_logsumexp = jnp.mean(jnp.stack(all_logsumexp, axis=0), axis=0)
            avg_correct = jnp.mean(jnp.stack(all_correct, axis=0), axis=0)
            avg_logits_pos = jnp.mean(jnp.array(all_logits_pos))
            avg_logits_neg = jnp.mean(jnp.array(all_logits_neg))
            
            return avg_loss, (avg_logsumexp, avg_correct, avg_logits_pos, avg_logits_neg)
        
        (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
            ensemble_loss, has_aux=True
        )(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)
        logsumexp_mean = logsumexp.mean()
    else:
        # Single critic case (original implementation)
        def critic_loss(critic_params, transitions, key):
            return single_critic_loss(
                critic_params["sa_encoder"], 
                critic_params["g_encoder"], 
                transitions, 
                key
            )

        (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
            critic_loss, has_aux=True
        )(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)
        logsumexp_mean = logsumexp.mean()

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp_mean,
        "critic_loss": loss,
    }

    return training_state, metrics
