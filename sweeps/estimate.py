"""Runtime model (A100-80GB, 1 GPU) for packing runs into jobs. Seconds per env step = a_env + b_method * depth.
Fitted on measured steady-state training speeds (gate3, smoke); +15% for evaluation, compilation, checkpoints.
Update the constants when new measurements arrive (see reports/gate3.md).
"""

import re

# simulator + data cost per env step (s), by env, at 512 envs
A_ENV = {"humanoid": 16.3e-6, "ant_big_maze": 25e-6, "arm_push_easy": 265e-6}  # arm: gate2 end-to-end 2.87k steps/s at d4
# network cost per env step per layer (s): measured SAC full humanoid d4 18.7k sps, d16 6.05k, d64 1.66k
B = {"sac": 9.3e-6, "td3": 9.3e-6, "crl": 6.5e-6}
OVERHEAD = 1.15
NAME = re.compile(r"^(sac|td3|crl)_.+_d(\d+)_(.+)_s\d+$")


def hours(name, total_steps=100_359_168):
    m = NAME.match(name)
    if not m:
        return 4.0
    method, depth, env = m.group(1), int(m.group(2)), m.group(3)
    a = A_ENV.get(env, 30e-6)
    return OVERHEAD * total_steps * (a + B[method] * depth) / 3600.0


def pack(names, max_hours):
    """First-fit decreasing: group runs so each group's estimated hours <= max_hours."""
    items = sorted(names, key=hours, reverse=True)
    groups = []
    for n in items:
        h = hours(n)
        for g in groups:
            if g[0] + h <= max_hours:
                g[0] += h
                g[1].append(n)
                break
        else:
            groups.append([h, [n]])
    # inside a job, run the shortest first, so results arrive early
    return [(h, sorted(g, key=hours)) for h, g in groups]
