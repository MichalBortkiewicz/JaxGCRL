"""Shared host-side evaluation, progress reporting, and checkpoint writing."""


class HostServices:
    def __init__(
        self,
        evaluator,
        progress_fn,
        unwrapped_env,
        visualization_interval,
        *,
        render_positionally=False,
    ):
        if visualization_interval < 1:
            raise ValueError("visualization_interval must be positive")
        self.evaluator = evaluator
        self.progress_fn = progress_fn
        self.unwrapped_env = unwrapped_env
        self.visualization_interval = visualization_interval
        self.render_positionally = render_positionally

    def publish(
        self,
        step,
        evaluation_state,
        make_policy,
        inference_params,
        training_metrics,
        *,
        epoch=None,
    ):
        metrics = self.evaluator.run_evaluation(evaluation_state, training_metrics)
        args = (step, metrics, make_policy, inference_params, self.unwrapped_env)
        if epoch is None:
            self.progress_fn(*args)
        elif self.render_positionally:
            self.progress_fn(*args, epoch % self.visualization_interval == 0)
        else:
            self.progress_fn(*args, do_render=epoch % self.visualization_interval == 0)
        return metrics

    @staticmethod
    def save(path, params, writer):
        writer(path, params)
