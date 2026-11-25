"""HostServices callback arity, render cadence, and error propagation tests."""

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_services_module():
    path = ROOT / "jaxgcrl/training/services.py"
    spec = importlib.util.spec_from_file_location("training_services", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strategy_services(strategy, module, callback):
    path = ROOT / f"jaxgcrl/training/{strategy}.py"
    tree = ast.parse(path.read_text())
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "HostServices"
    )

    class Evaluation:
        def run_evaluation(self, params, metrics):
            del params
            return metrics

    class Config:
        visualization_interval = 3

    return eval(
        compile(ast.Expression(call), str(path), "eval"),
        {
            "HostServices": module.HostServices,
            "evaluator": Evaluation(),
            "progress_fn": callback,
            "unwrapped_env": "env",
            "config": Config(),
        },
    )


class CallbackTests(unittest.TestCase):
    def test_shared_publication_preserves_payload_and_render_cadence(self):
        module = _load_services_module()
        events = []

        class Evaluation:
            def run_evaluation(self, params, metrics):
                events.append(("evaluate", params, metrics))
                return {**metrics, "eval/reward": 3.0}

        def policy(params):
            del params
            return lambda obs, key: (obs, {})

        def callback(*args, **kwargs):
            events.append(("callback", args, kwargs))

        services = module.HostServices(Evaluation(), callback, "env", 3)
        result = services.publish(8, "eval-state", policy, "actor-params", {"training/loss": 2.0}, epoch=1)
        self.assertEqual(result, {"training/loss": 2.0, "eval/reward": 3.0})
        self.assertEqual(events[0], ("evaluate", "eval-state", {"training/loss": 2.0}))
        self.assertEqual(events[1][1], (8, result, policy, "actor-params", "env"))
        self.assertEqual(events[1][2], {"do_render": False})

        services.publish(16, "eval-state", policy, "actor-params", {}, epoch=3)
        self.assertEqual(events[-1][2], {"do_render": True})

        services.publish(0, "eval-state", policy, "actor-params", {})
        self.assertEqual(events[-1][2], {})

    def test_off_policy_positional_callbacks_and_initial_arity(self):
        module = _load_services_module()
        events = []

        def args_only(*args):
            events.append(args)

        services = _strategy_services("off_policy", module, args_only)
        services.publish(0, None, None, None, {})
        self.assertEqual(len(events[-1]), 5)
        for epoch, expected in [(1, False), (3, True)]:
            services.publish(8, None, None, None, {}, epoch=epoch)
            self.assertEqual(len(events[-1]), 6)
            self.assertIs(events[-1][-1], expected)

        def renamed(step, metrics, policy, params, env, render_now):
            del step, metrics, policy, params, env
            events.append(render_now)

        services.progress_fn = renamed
        services.publish(8, None, None, None, {}, epoch=3)
        self.assertIs(events[-1], True)

    def test_crl_and_ppo_keep_keyword_render_and_initial_arity(self):
        module = _load_services_module()
        for strategy in ("episode_off_policy", "on_policy"):
            events = []

            def callback(step, metrics, policy, params, env, _events=events, *, do_render=None):
                del step, metrics, policy, params, env
                _events.append(do_render)

            services = _strategy_services(strategy, module, callback)
            services.publish(0, None, None, None, {})
            services.publish(8, None, None, None, {}, epoch=1)
            services.publish(16, None, None, None, {}, epoch=3)
            self.assertEqual(events, [None, False, True])

    def test_callback_type_error_is_not_retried(self):
        module = _load_services_module()

        class Evaluation:
            def run_evaluation(self, params, metrics):
                del params
                return metrics

        calls = []

        def callback(*args, **kwargs):
            calls.append((args, kwargs))
            raise TypeError("inside callback")

        services = module.HostServices(Evaluation(), callback, "env", 3)
        with self.assertRaisesRegex(TypeError, "inside callback"):
            services.publish(8, None, None, None, {}, epoch=1)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
