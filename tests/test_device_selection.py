"""Device selection, collective participant ordering, and replica placement tests."""

import ast
import importlib.util
import sys
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Device:
    process_index: int
    id: int


@contextmanager
def _contracts(counts=(2, 2), rank=0):
    hosts = [tuple(Device(p, i) for i in range(n)) for p, n in enumerate(counts)]
    fake: Any = ModuleType("jax")
    fake.local_devices = lambda: list(hosts[rank])
    fake.devices = lambda: [d for host in hosts for d in host]
    fake.process_count = lambda: len(counts)
    fake.process_index = lambda: rank
    fake.random = SimpleNamespace(
        split=lambda key, n: [(key, i) for i in range(n)],
        fold_in=lambda key, r: ("fold", key, r),
    )
    fake.tree_util = SimpleNamespace(tree_map=lambda fn, x: fn(x))
    numpy: Any = ModuleType("jax.numpy")
    numpy.mean = lambda x: x
    with patch.dict(sys.modules, {"jax": fake, "jax.numpy": numpy}):
        spec = importlib.util.spec_from_file_location(
            "_device_contracts", ROOT / "jaxgcrl/training/contracts.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"_device_contracts": module}):
            spec.loader.exec_module(module)
            yield module, fake, hosts


class DeviceSelectionTests(unittest.TestCase):
    def test_update_uses_global_participants_but_local_state_and_keys(self):
        participant_lists = []
        key_batches = []
        for rank in (0, 1):
            with _contracts(rank=rank) as (module, fake, hosts):
                local = module.selected_devices(SimpleNamespace(max_devices_per_host=1))
                self.assertEqual(local, hosts[rank][:1])
                state = object()
                batch = object()

                def pmap(
                    fn,
                    *,
                    axis_name,
                    devices,
                    _state=state,
                    _batch=batch,
                    _local=local,
                ):
                    del fn, axis_name
                    participant_lists.append(devices)

                    def apply(actual_state, actual_batch, keys):
                        self.assertIs(actual_state, _state)
                        self.assertIs(actual_batch, _batch)
                        self.assertEqual(len(keys), len(_local))
                        key_batches.append(keys)
                        return (_state, None), 7

                    return apply

                fake.pmap = pmap
                learner = module.Learner(None, state, None, None, None, local)
                updated, metrics = module.update_minibatch(learner, batch, "key")
                self.assertIs(updated.state, state)
                self.assertEqual(updated.devices, local)
                self.assertEqual(metrics, 7)

        expected = (Device(0, 0), Device(1, 0))
        self.assertEqual(participant_lists, [expected, expected])
        self.assertNotEqual(key_batches[0], key_batches[1])

    def test_single_process_keeps_exact_key_split_and_device_tuple(self):
        with _contracts((3,)) as (module, _, hosts):
            local = module.selected_devices(SimpleNamespace(max_devices_per_host=2))
            self.assertEqual(local, hosts[0][:2])
            self.assertIs(module.collective_devices(local), local)
            self.assertEqual(module.local_update_keys("key", 2), [("key", 0), ("key", 1)])

    def test_global_order_is_process_then_local_order_on_every_host(self):
        for rank in (0, 1):
            with _contracts(rank=rank) as (module, fake, hosts):
                fake.devices = lambda: [hosts[1][0], hosts[0][0], hosts[1][1], hosts[0][1]]
                local = module.selected_devices(SimpleNamespace(max_devices_per_host=1))
                self.assertEqual(module.collective_devices(local), (hosts[0][0], hosts[1][0]))

    def test_every_collection_pmap_uses_global_participants(self):
        paths = [
            "jaxgcrl/training/off_policy.py",
            "jaxgcrl/training/on_policy.py",
            "jaxgcrl/agents/sac/sac.py",
        ]
        for rank in (0, 1):
            with _contracts(rank=rank) as (module, fake, hosts):
                local = hosts[rank][:1]
                expected = (hosts[0][0], hosts[1][0])
                calls = []
                fake.pmap = lambda *args, _calls=calls, **kwargs: (
                    _calls.append(kwargs["devices"]) or (lambda *_a, **_k: None)
                )
                for path in paths:
                    tree = ast.parse((ROOT / path).read_text())
                    nodes = [
                        n
                        for n in ast.walk(tree)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "pmap"
                        and isinstance(n.func.value, ast.Name)
                        and n.func.value.id == "jax"
                    ]
                    self.assertTrue(nodes, path)
                    for node in nodes:
                        namespace = {
                            "jax": fake,
                            "devices": local,
                            "learner": SimpleNamespace(devices=local),
                            "collective_devices": module.collective_devices,
                            "env": SimpleNamespace(reset=None),
                            "replay_buffer": SimpleNamespace(init=None),
                            "prefill_replay_buffer": None,
                            "training_epoch": None,
                            "local_devices_to_use": len(local),
                            "rb_key": "rb_key",
                            "keys": "keys",
                            "_PMAP_AXIS_NAME": "i",
                        }
                        eval(compile(ast.Expression(node), path, "eval"), namespace)
                        self.assertEqual(calls[-1], expected, path)

    def test_agent_initialization_keeps_local_placement(self):
        for rank in (0, 1):
            with _contracts(rank=rank) as (module, fake, hosts):
                local = module.selected_devices(SimpleNamespace(max_devices_per_host=1))
                placements = []
                fake.device_put_replicated = lambda state, devices, _placements=placements: (
                    _placements.append(tuple(devices))
                )
                for name in ("sac", "td3", "ppo"):
                    path = ROOT / f"jaxgcrl/agents/{name}/{name}.py"
                    tree = ast.parse(path.read_text())
                    node = next(
                        n
                        for n in ast.walk(tree)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "device_put_replicated"
                    )
                    eval(
                        compile(ast.Expression(node), str(path), "eval"),
                        {
                            "jax": fake,
                            "training_state": object(),
                            "local_devices_to_use": len(local),
                        },
                    )
                    self.assertEqual(placements[-1], local, name)

    def test_invalid_caps_empty_and_unequal_topologies_are_rejected(self):
        for cap in (0, -1):
            with _contracts() as (module, _, _):
                with self.assertRaises(ValueError):
                    module.selected_devices(SimpleNamespace(max_devices_per_host=cap))
        for counts in ((0,), (2, 1), (2, 0)):
            with _contracts(counts) as (module, _, _):
                with self.assertRaises(ValueError):
                    module.selected_devices(SimpleNamespace(max_devices_per_host=1))


if __name__ == "__main__":
    unittest.main()
