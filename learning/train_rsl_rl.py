# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=wrong-import-position
"""Train a PPO agent using RSL-RL for the specified environment."""

import os
import re

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

from datetime import datetime
import json

from absl import app
from absl import flags
from absl import logging
import jax
import mediapy as media
from ml_collections import config_dict
import mujoco
import onnx
from onnx import external_data_helper
from rsl_rl.runners import OnPolicyRunner
import torch
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

# Suppress logs if you want
logging.set_verbosity(logging.WARNING)

# Define flags similar to the JAX script
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "Go2JoystickFlatTerrain",
    (
        "Name of the environment. One of: "
        f"{', '.join(mujoco_playground.registry.ALL_ENVS)}"
    ),
)
_LOAD_RUN_NAME = flags.DEFINE_string(
    "load_run_name", None, "Run name to load from (for checkpoint restoration)."
)
_CHECKPOINT_NUM = flags.DEFINE_integer(
    "checkpoint_num", -1, "Checkpoint number to load from."
)
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train."
)
_EXPORT_ONNX_ONLY = flags.DEFINE_boolean(
    "export_onnx_only",
    False,
    (
        "If true, skip training/playback and export <run_name>.onnx from an "
        "existing checkpoint. Uses --load_run_name, or the latest run matching "
        "--env_name when --load_run_name is unset."
    ),
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    True,
    "Use Weights & Biases for logging (ignored in play-only mode).",
)
_WANDB_ENTITY = flags.DEFINE_string(
    "wandb_entity",
    None,
    "Weights & Biases team/entity to log runs under.",
)
_WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project",
    None,
    "Weights & Biases project to log runs under.",
)
_WANDB_RUN_NAME = flags.DEFINE_string(
    "wandb_run_name",
    None,
    (
        "Optional run name for both the local logs/<run_name> directory and "
        "the W&B run name. Defaults to the timestamped experiment name."
    ),
)
flags.DEFINE_alias("wandb-run-name", "wandb_run_name")
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name.")
_SEED = flags.DEFINE_integer("seed", 42, "Random seed.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 2048, "Number of parallel envs.")
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for training.")
_MULTI_GPU = flags.DEFINE_boolean(
    "multi_gpu", False, "If true, use multi-GPU training (distributed)."
)
_CAMERA = flags.DEFINE_string(
    "camera", None, "Camera name to use for rendering."
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in registry.manipulation._envs:
    return manipulation_params.rsl_rl_config(env_name)
  elif env_name in registry.locomotion._envs:
    return locomotion_params.rsl_rl_config(env_name)
  else:
    raise ValueError(f"No RL config for {env_name}")


def _latest_model_checkpoint(logdir: str) -> str:
  """Returns the latest RSL-RL model checkpoint under a training run directory."""
  candidates = []
  for root, _, files in os.walk(logdir):
    for filename in files:
      if filename.startswith("model_") and filename.endswith(".pt"):
        candidates.append(os.path.join(root, filename))
  if not candidates:
    raise FileNotFoundError(f"No model_*.pt checkpoint found in {logdir}")

  def sort_key(path: str) -> tuple[int, float, str]:
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
    step = int(match.group(1)) if match else -1
    return step, os.path.getmtime(path), path

  return max(candidates, key=sort_key)


def _checkpoint_path(logdir: str, checkpoint_num: int = -1) -> str:
  """Returns a requested checkpoint, or the latest checkpoint when unspecified."""
  if checkpoint_num == -1:
    return _latest_model_checkpoint(logdir)

  checkpoint_path = os.path.join(logdir, f"model_{checkpoint_num}.pt")
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
  return checkpoint_path


def _resolve_checkpoint_logdir(
    root: str, env_name: str, load_run_name: str | None
) -> str:
  """Resolves the run directory to use for checkpoint-only workflows."""
  if load_run_name:
    if os.path.isabs(load_run_name):
      logdir = load_run_name
    else:
      logdir = os.path.join(root, load_run_name)
    if not os.path.isdir(logdir):
      raise FileNotFoundError(f"Run directory not found: {logdir}")
    return logdir

  candidates = []
  if os.path.isdir(root):
    for run_name in os.listdir(root):
      run_dir = os.path.join(root, run_name)
      if run_name.startswith(f"{env_name}-") and os.path.isdir(run_dir):
        try:
          checkpoint = _latest_model_checkpoint(run_dir)
        except FileNotFoundError:
          continue
        candidates.append((os.path.getmtime(checkpoint), run_dir))

  if not candidates:
    raise FileNotFoundError(
        f"No checkpointed runs found for env '{env_name}' under {root}"
    )
  return max(candidates)[1]


def _external_data_locations(model: onnx.ModelProto) -> set[str]:
  """Returns sidecar filenames referenced by an ONNX model."""
  locations = set()
  for tensor in external_data_helper._get_all_tensors(model):  # pylint: disable=protected-access
    if external_data_helper.uses_external_data(tensor):
      for entry in tensor.external_data:
        if entry.key == "location":
          locations.add(entry.value)
  return locations


def _make_onnx_self_contained(onnx_path: str) -> None:
  """Embeds external tensor data so the policy is a single .onnx file."""
  model = onnx.load_model(onnx_path, load_external_data=False)
  external_locations = _external_data_locations(model)
  if not external_locations:
    return

  base_dir = os.path.dirname(onnx_path)
  external_data_helper.load_external_data_for_model(model, base_dir)
  onnx.save_model(model, onnx_path, save_as_external_data=False)

  for location in external_locations:
    external_path = os.path.abspath(os.path.join(base_dir, location))
    if external_path == os.path.abspath(onnx_path):
      continue
    if os.path.exists(external_path):
      os.remove(external_path)


def _onnx_filename_from_run_name(run_name: str) -> str:
  """Returns a filesystem-safe ONNX filename based on the run name."""
  safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("._")
  if not safe_run_name:
    safe_run_name = "policy"
  return f"{safe_run_name}.onnx"


def _export_latest_policy_onnx(
    runner,
    logdir: str,
    device: str,
    run_name: str,
    checkpoint_num: int = -1,
    save_if_missing: bool = True,
) -> str:
  """Loads the latest saved checkpoint and exports it as <run_name>.onnx."""
  del device  # RSL-RL exports ONNX from a CPU copy of the policy.
  try:
    latest_checkpoint = _checkpoint_path(logdir, checkpoint_num)
  except FileNotFoundError:
    if not save_if_missing:
      raise
    latest_checkpoint = os.path.join(
        logdir, f"model_{runner.current_learning_iteration}.pt"
    )
    print(f"No saved checkpoint found; saving final checkpoint: {latest_checkpoint}")
    runner.save(latest_checkpoint)
  output_dir = os.path.dirname(latest_checkpoint)
  onnx_filename = _onnx_filename_from_run_name(run_name)
  output_path = os.path.join(output_dir, onnx_filename)

  print(f"Loading latest checkpoint for ONNX export: {latest_checkpoint}")
  # RSL-RL may update observation normalizer buffers while collecting rollouts
  # under torch.inference_mode(), which makes them inference tensors. Loading
  # and exporting such buffers must also happen in inference mode.
  with torch.inference_mode():
    runner.load(latest_checkpoint)
    runner.export_policy_to_onnx(output_dir, filename=onnx_filename)
    _make_onnx_self_contained(output_path)

  print(f"Exported ONNX policy: {output_path}")
  return output_path


def _render_camera(env, requested_camera: str | None) -> str | None:
  """Returns the requested camera, or a tracking camera when available."""
  if requested_camera is not None:
    return requested_camera

  if mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track") >= 0:
    return "track"
  return None


def main(argv):
  """Run training and evaluation for the specified environment using RSL-RL."""
  del argv  # unused

  # Possibly parse the device for multi-GPU
  if _MULTI_GPU.value:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_rank = local_rank
    device = f"cuda:{local_rank}"
    print(f"Using multi-GPU: local_rank={local_rank}, device={device}")
  else:
    device = _DEVICE.value
    device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

  # If play-only or export-only, use fewer envs
  num_envs = 1 if _PLAY_ONLY.value or _EXPORT_ONNX_ONLY.value else _NUM_ENVS.value

  # Load default config from registry
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  print(f"Environment config:\n{env_cfg}")

  if _PLAY_ONLY.value or _EXPORT_ONNX_ONLY.value:
    logdir = _resolve_checkpoint_logdir(
        os.path.abspath("logs"), _ENV_NAME.value, _LOAD_RUN_NAME.value
    )
    exp_name = os.path.basename(logdir)
    print(f"Loading checkpointed run directory: {logdir}")
  else:
    # Generate or use an explicit experiment name. RSL-RL's WandbLogWriter
    # derives the online run name from the local log directory basename.
    if _WANDB_RUN_NAME.value is not None:
      now = datetime.now()
      timestamp = now.strftime("%Y%m%d-%H%M")
      exp_name = f"{_WANDB_RUN_NAME.value}-{timestamp}"

    else:
      now = datetime.now()
      timestamp = now.strftime("%Y%m%d-%H%M")
      exp_name = f"{_ENV_NAME.value}-{timestamp}"
      if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment name: {exp_name}")

    # Logging directory
    logdir = os.path.abspath(os.path.join("logs", exp_name))
    os.makedirs(logdir, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Checkpoint directory
    ckpt_path = os.path.join(logdir, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

  if not _PLAY_ONLY.value and not _EXPORT_ONNX_ONLY.value:
    # Save environment config to JSON
    with open(
        os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8"
    ) as fp:
      json.dump(env_cfg.to_dict(), fp, indent=4)

  # Domain randomization
  randomizer = registry.get_domain_randomizer(_ENV_NAME.value)

  # We'll store environment states during rendering
  render_trajectory = []

  # Callback to gather states for rendering
  def render_callback(_, state):
    render_trajectory.append(state)

  # Create the environment
  raw_env = registry.load(_ENV_NAME.value, config=env_cfg)
  brax_env = wrapper_torch.RSLRLBraxWrapper(
      raw_env,
      num_envs,
      _SEED.value,
      env_cfg.episode_length,
      1,
      render_callback=render_callback,
      randomization_fn=randomizer,
      device_rank=device_rank,
  )

  # Build RSL-RL config
  train_cfg = get_rl_config(_ENV_NAME.value)
  obs_size = raw_env.observation_size
  if isinstance(obs_size, dict):
    train_cfg.obs_groups = {"actor": ["state"], "critic": ["privileged_state"]}
  else:
    train_cfg.obs_groups = {"actor": ["state"], "critic": ["state"]}

  # Overwrite default config with flags
  train_cfg.seed = _SEED.value
  train_cfg.run_name = exp_name
  train_cfg.resume = (
      (_LOAD_RUN_NAME.value is not None or _PLAY_ONLY.value)
      and not _EXPORT_ONNX_ONLY.value
  )
  train_cfg.load_run = _LOAD_RUN_NAME.value if _LOAD_RUN_NAME.value else "-1"
  train_cfg.checkpoint = _CHECKPOINT_NUM.value
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if _WANDB_ENTITY.value:
      # RSL-RL's WandbLogWriter reads the W&B entity from WANDB_USERNAME.
      os.environ["WANDB_USERNAME"] = _WANDB_ENTITY.value
    train_cfg.logger = {
        "class_name": "WandbLogWriter",
        "project_name": _WANDB_PROJECT.value,
    }

  train_cfg_dict = train_cfg.to_dict()
  runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)

  if _EXPORT_ONNX_ONLY.value:
    _export_latest_policy_onnx(
        runner,
        logdir,
        device,
        exp_name,
        checkpoint_num=_CHECKPOINT_NUM.value,
        save_if_missing=False,
    )
    return

  # If resume, load from checkpoint
  if train_cfg.resume:
    if _PLAY_ONLY.value:
      resume_path = _checkpoint_path(logdir, _CHECKPOINT_NUM.value)
    else:
      resume_path = wrapper_torch.get_load_path(
          os.path.abspath("logs"),
          load_run=train_cfg.load_run,
          checkpoint=train_cfg.checkpoint,
      )
    print(f"Loading model from checkpoint: {resume_path}")
    runner.load(resume_path)

  if not _PLAY_ONLY.value:
    # Perform training
    runner.learn(
        num_learning_iterations=train_cfg.max_iterations,
        init_at_random_ep_len=False,
    )
    print("Done training.")
    _export_latest_policy_onnx(runner, logdir, device, exp_name)
    return

  # If just playing (no training)
  policy = runner.get_inference_policy(device=device)

  # Example: run a single rollout
  eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(_SEED.value)
  state = jit_reset(rng)
  rollout = [state]

  is_dict_obs = isinstance(eval_env.observation_size, dict)
  obs = state.obs["state"] if is_dict_obs else state.obs
  obs_torch = wrapper_torch._jax_to_torch(obs)

  for _ in range(env_cfg.episode_length):
    with torch.no_grad():
      actions = policy({"state": obs_torch})
      actions = torch.clip(actions, -1.0, 1.0)
    # Step environment
    state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
    rollout.append(state)
    obs = state.obs["state"] if is_dict_obs else state.obs
    obs_torch = wrapper_torch._jax_to_torch(obs)
    if state.done:
      break

  # Render
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  render_every = 2
  # If your environment is wrapped multiple times, adjust as needed:
  base_env = eval_env  # or brax_env.env.env.env
  fps = 1.0 / base_env.dt / render_every
  traj = rollout[::render_every]
  frames = eval_env.render(
      traj,
      camera=_render_camera(eval_env, _CAMERA.value),
      height=480,
      width=640,
      scene_option=scene_option,
  )
  rollout_path = os.path.join(logdir, "rollout.mp4")
  media.write_video(rollout_path, frames, fps=fps)
  print(f"Rollout video saved as: {rollout_path}")


def run():
  """Entry point for the train-rsl-ppo console script."""
  app.run(main)


if __name__ == "__main__":
  run()