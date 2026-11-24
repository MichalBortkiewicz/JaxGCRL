"""Run definitions for every experiment set. Writes scripts/<set>/runs.tsv (name <TAB> method <TAB> flags).

Matched settings (03_EXPERIMENTS.md §1): 512 envs, unroll 78, batch 512, gamma 0.99, lr 3e-4, replay 10k/1k
per env, width 256. One training step = 512 x 78 = 39,936 env steps and ~1000 gradient steps (UTD ~1/40).
Step budgets are expressed as `epochs` x `steps_per_epoch` training steps so that SAC/TD3 (eval at step 0 +
one per epoch) and CRL (one eval per epoch) see exactly the same number of env steps and gradient steps:
    total_env_steps = 13 * U + epochs * steps_per_epoch * U,   U = 39,936   (13 prefill steps of 78)
usage: python sweeps/matrix.py <set>    (prints the runs and writes scripts/<set>/runs.tsv)
"""

import itertools
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_ENVS, UNROLL = 512, 78
U = NUM_ENVS * UNROLL
PREFILL_STEPS = 13  # SAC/TD3: min_replay // unroll + 1 = 13; CRL: ceil(min_replay / unroll) = 13

VARIANT_FLAGS = {
    "plain": "--no-use-residual --no-use-ln --activation relu",
    "full": "--use-residual --use-ln --activation swish",
    "full_no_ln": "--use-residual --no-use-ln --activation swish",
    "full_no_res": "--no-use-residual --use-ln --activation swish",
    "full_no_swish": "--use-residual --use-ln --activation relu",
}
ENVS1 = ["humanoid", "ant_big_maze", "arm_push_easy"]
# The 1000-layer paper's other environments that exist in JaxGCRL (its ant_u4_maze / ant_u5_maze do not).
ENVS_EXTRA = ["ant_hardest_maze", "humanoid_u_maze", "humanoid_big_maze", "arm_push_hard", "arm_binpick_hard"]
DEPTHS = [4, 8, 16, 32, 64]


def budget_flags(method, epochs, steps_per_epoch):
    total = PREFILL_STEPS * U + epochs * steps_per_epoch * U
    n_evals = epochs + 1 if method in ("sac", "td3") else epochs
    return f"--total-env-steps {total} --num-evals {n_evals}"


def matched_common():
    return (f"--num-envs {NUM_ENVS} --episode-length 1001 --unroll-length {UNROLL} --batch-size 512 "
            f"--train-step-multiplier 1 --discounting 0.99 --max-replay-size 10000 --min-replay-size 1000 --h-dim 256")


def td_run(method, variant, depth, env, seed, epochs, spe, her="future", tag=None):
    flags = (f"{method} --env {env} --seed {seed} {matched_common()} --learning-rate 3e-4 "
             f"--network-type scaled --n-hidden {depth} {VARIANT_FLAGS[variant]} --use-her --her-strategy {her} "
             f"--her-ratio 0.8 {budget_flags(method, epochs, spe)}")
    if method == "td3":
        flags += " --mask-truncation --fix-target-policy"
    name = f"{method}_{variant}{'_' + tag if tag else ''}_d{depth}_{env}_s{seed}"
    return name, method, flags


def crl_run(depth, env, seed, epochs, spe):
    flags = (f"crl --env {env} --seed {seed} {matched_common()} --network-type scaled --n-hidden {depth} "
             f"{budget_flags('crl', epochs, spe)}")
    return f"crl_scaled_d{depth}_{env}_s{seed}", "crl", flags


def legacy_sac_run(depth, env, seed, total=50_000_000, n_evals=51):
    # Arm A0: unmodified JaxGCRL defaults (gamma 0.9, lr 1e-4, batch 256, 256 envs, unroll 50, final HER, ReLU)
    return (f"sac_legacy_d{depth}_{env}_s{seed}", "sac",
            f"sac --env {env} --seed {seed} --use-her --n-hidden {depth} --total-env-steps {total} --num-evals {n_evals}")


def runs_for(set_name):
    R = []
    if set_name == "smoke":
        # one tiny run per method x env (smoke + resume test); full variant, depth 8
        for env in ENVS1:
            R.append(td_run("sac", "full", 8, env, 0, epochs=3, spe=8))
            R.append(td_run("td3", "full", 8, env, 0, epochs=3, spe=8))
            R.append(crl_run(8, env, 0, epochs=3, spe=8))
        R.append(legacy_sac_run(4, "ant_big_maze", 0, total=1_500_000, n_evals=3))
    elif set_name == "gate1":
        for d, s in itertools.product([4, 8], [0, 1, 2]):
            R.append(legacy_sac_run(d, "ant_big_maze", s))
    elif set_name == "gate2":
        env = "arm_push_easy"
        for s in [0, 1]:
            # legacy network with matched hyperparameters, final HER
            n, m, f = td_run("sac", "plain", 4, env, s, epochs=20, spe=25, her="final", tag="final")
            f = f.replace("--network-type scaled", "--network-type legacy").replace(VARIANT_FLAGS["plain"], "")
            R.append((f"sac_legacymlp_final_d4_{env}_s{s}", m, f))
            R.append(td_run("sac", "plain", 4, env, s, epochs=20, spe=25, her="final", tag="final"))
            R.append(td_run("sac", "plain", 4, env, s, epochs=20, spe=25))
            R.append(td_run("sac", "full", 4, env, s, epochs=20, spe=25))
            R.append(td_run("td3", "full", 4, env, s, epochs=20, spe=25))
    elif set_name == "gate3":
        # timing / memory: 2 epochs of 25 training steps (~2.5M env steps)
        for d in [4, 16, 64]:
            R.append(td_run("sac", "full", d, "humanoid", 0, epochs=2, spe=25))
        R.append(td_run("td3", "full", 64, "humanoid", 0, epochs=2, spe=25))
        R.append(crl_run(64, "humanoid", 0, epochs=2, spe=25))
        R.append(td_run("sac", "full", 64, "arm_push_easy", 0, epochs=2, spe=25))
        R.append(td_run("sac", "full", 64, "ant_big_maze", 0, epochs=2, spe=25))
    elif set_name == "td3diag":
        # Why does TD3's actor saturate on humanoid? depth 4, 20.5M steps, seed 0, one factor at a time.
        env = "humanoid"
        base = td_run("td3", "plain", 4, env, 0, epochs=20, spe=25)[2]
        R.append(("td3diag_matched_d4_humanoid_s0", "td3", base))  # = stage1 config (both fixes on)
        R.append(("td3diag_nofixtarget_d4_humanoid_s0", "td3", base.replace(" --fix-target-policy", "")))
        R.append(("td3diag_nomask_d4_humanoid_s0", "td3", base.replace(" --mask-truncation", "")))
        R.append(("td3diag_nofixes_d4_humanoid_s0", "td3",
                  base.replace(" --mask-truncation", "").replace(" --fix-target-policy", "")))
        R.append(("td3diag_gamma09_d4_humanoid_s0", "td3", base.replace("--discounting 0.99", "--discounting 0.9")))
        # unmodified JaxGCRL TD3 (2x256 ReLU, gamma 0.9, final HER, 256 envs, unroll 50)
        R.append(("td3diag_legacy_d4_humanoid_s0", "td3",
                  f"td3 --env {env} --seed 0 --use-her --total-env-steps 20000000 --num-evals 21"))
        R.append(("sacdiag_matched_d4_humanoid_s0", "sac", td_run("sac", "plain", 4, env, 0, epochs=20, spe=25)[2]))
    elif set_name == "lac1":
        # LAC follow-up (reports/methods.md, pre-registered 2026-11-24): SAC full, light actor (depth 4),
        # 2 x 2 of n-step {1, 4} x critic loss {mse, categorical}; depths 4/16/64; dev envs where depth can
        # show (humanoid, ant_big_maze); seeds 0-2; 100M steps. Plus Stage 1 baseline (deep actor) seed 2.
        arms = {
            "la": "--actor-n-hidden 4",
            "la_n4": "--actor-n-hidden 4 --n-step 4",
            "la_cat": "--actor-n-hidden 4 --critic-loss categorical",
            "la_n4_cat": "--actor-n-hidden 4 --n-step 4 --critic-loss categorical",
        }
        for s, d, env in itertools.product([0, 1, 2], [4, 16, 64], ["ant_big_maze", "humanoid"]):
            for tag, extra in arms.items():
                if tag == "la" and d == 4:
                    continue  # identical to the Stage 1 baseline (actor depth = critic depth = 4)
                n, m, f = td_run("sac", "full", d, env, s, 100, 25, tag=tag)
                R.append((n, m, f + " " + extra))
            if s == 2:
                R.append(td_run("sac", "full", d, env, s, 100, 25))
    elif set_name == "lacsmoke":
        # short GPU check of the categorical critic + n-step before lac1
        cat4 = "--actor-n-hidden 4 --n-step 4 --critic-loss categorical"
        n, m, f = td_run("sac", "full", 16, "ant_big_maze", 0, epochs=3, spe=8, tag="la_n4_cat")
        R.append((n, m, f + " " + cat4))
        n, m, f = td_run("sac", "full", 4, "humanoid", 0, epochs=3, spe=8, tag="la_n4_catdist")
        R.append((n, m, f + " " + cat4 + " --cat-projection distributional --critic-head-ln"))
        n, m, f = td_run("td3", "full", 16, "arm_push_easy", 0, epochs=3, spe=8, tag="la_n4_cat")
        R.append((n, m, f + " " + cat4))
    elif set_name == "repsmoke":
        # check the representation diagnostics (rep_metrics.py) appear in metrics.csv in real runs
        R.append(td_run("sac", "full", 16, "ant_big_maze", 0, epochs=3, spe=8))
        R.append(td_run("sac", "plain", 16, "humanoid", 0, epochs=3, spe=8))
        R.append(td_run("td3", "full", 16, "arm_push_easy", 0, epochs=3, spe=8))
        R.append(legacy_sac_run(4, "ant_big_maze", 0, total=3_000_000, n_evals=3))
    elif set_name == "extra_smoke":
        # smoke + resume test for the extra environments (Jubail handoff): sac and td3, full, depth 8
        for env in ENVS_EXTRA:
            R.append(td_run("sac", "full", 8, env, 0, epochs=3, spe=8))
            R.append(td_run("td3", "full", 8, env, 0, epochs=3, spe=8))
    elif set_name == "extra":
        # Stage 1 design on the extra environments: SAC/TD3 x plain/full x 5 depths x 5 envs x seeds 0, 1
        arms = [("sac", "plain"), ("sac", "full"), ("td3", "plain"), ("td3", "full")]
        for s, d, env, (m, v) in itertools.product([0, 1], DEPTHS, ENVS_EXTRA, arms):
            R.append(td_run(m, v, d, env, s, 100, 25))
    elif set_name in ("stage1", "stage2", "stage3"):
        if set_name == "stage1":
            # user 2026-11-20: no CRL (compare with the paper's CRL figures); SAC and TD3, plain and full
            arms = [("sac", "plain"), ("sac", "full"), ("td3", "plain"), ("td3", "full")]
        elif set_name == "stage2":
            arms = [("sac", v) for v in ("full_no_ln", "full_no_res", "full_no_swish")]
        else:
            arms = [("td3", v) for v in VARIANT_FLAGS]
        seeds = [0, 1] if set_name == "stage1" else [0, 1, 2]  # user 2026-11-20: seeds 0 and 1, seed 0 first
        for s, d, env, (m, v) in itertools.product(seeds, DEPTHS, ENVS1, arms):
            R.append(crl_run(d, env, s, 100, 25) if m == "crl" else td_run(m, v, d, env, s, 100, 25))
    else:
        raise SystemExit(f"unknown set {set_name}")
    common = f"--no-render --auto-resume --wandb-project-name scaling-sac --wandb-group {set_name}"
    if set_name in ("smoke", "extra_smoke"):
        common += " --ckpt-interval-min 0"  # checkpoint at every eval so the resume test has something to resume
    return [(n, m, f"{f} {common}") for n, m, f in R]


if __name__ == "__main__":
    set_name = sys.argv[1]
    runs = runs_for(set_name)
    out_dir = os.path.join(REPO, "scripts", set_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "runs.tsv"), "w") as f:
        for n, m, fl in runs:
            f.write(f"{n}\t{m}\t{fl}\n")
    for n, m, fl in runs:
        print(n)
    print(f"{len(runs)} runs -> scripts/{set_name}/runs.tsv", file=sys.stderr)
