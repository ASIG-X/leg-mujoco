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
"""Joystick task for Go2."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import base as go2_base
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=5000,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=10,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=1.0,
              # Base reward.
              height=1.0,
              lin_vel_z=-1.5,
              action_rate=-0.05,
              action=-0.1,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              pose=1.2,
              stand_still=-1.0,
              energy=-0.001,
          ),
          tracking_sigma=0.25,
          target_base_height=0.35,
      ),
      pert_config=config_dict.create(
          enable=True,
          velocity_kick=[-1.0, 1.0],
          push_interval_s=4.0,
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          vel_x=[-1.5, 1.5],
          vel_y=[-1.0, 1.0],
          yaw_rate=[-1.5, 1.5],
          zero_prob=[0.1, 0.7, 0.5],
          ctrl_interval=[60, 140],
      ),
  )


class Joystick(go2_base.Go2Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_lo = jp.array([
        self._config.command_config.vel_x[0],
        self._config.command_config.vel_y[0],
        self._config.command_config.yaw_rate[0],
    ])
    self._cmd_hi = jp.array([
        self._config.command_config.vel_x[1],
        self._config.command_config.vel_y[1],
        self._config.command_config.yaw_rate[1],
    ])
    self._cmd_zero_prob = jp.array(self._config.command_config.zero_prob)
    self._cmd_ctrl_interval = self._config.command_config.ctrl_interval
    
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)
    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])
    info = {
        "command": jp.zeros(3),
        "last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "ground_z": jp.zeros(()),
    }
    actor_obs = self._get_actor_obs(data, info)
    critic_obs = self._get_critic_obs(data, info, actor_obs)
    self._actor_frame_obs_dim = actor_obs.shape[-1]
    self._critic_obs_dim = critic_obs.shape[-1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # JAVE reset distribution: small xy, yaw, and joint perturbations.
    rng, key = jax.random.split(rng)
    dxy = 0.02 * (jax.random.uniform(key, (2,)) - 0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = (jp.pi / 20.0) * (jax.random.uniform(key, ()) - 0.5)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    qpos = qpos.at[3:7].set(quat)

    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].add(0.03 * (jax.random.uniform(key, (12,)) - 0.5))

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    push_interval_steps = jp.round(
        self._config.pert_config.push_interval_s / self.dt
    ).astype(
        jp.int32
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    steps_until_next_cmd = self._sample_ctrl_interval(key1)
    cmd = self.sample_command(key2)

    info = {
        "step": jp.array(0, dtype=jp.int32),
        "rng": rng,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "ground_z": jp.zeros(()),
        "actor_obs_history": jp.zeros(
            (self._config.history_len, self._actor_frame_obs_dim)
        ),
        "push_interval_steps": push_interval_steps,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    actor_frame = self._get_actor_obs(data, info)
    info["actor_obs_history"] = jp.repeat(
        actor_frame[None, :], self._config.history_len, axis=0
    )
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    # state = self._reset_if_outside_bounds(state)

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] += 1
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1),
        state.info["command"],
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        self._sample_ctrl_interval(key2),
        state.info["steps_until_next_cmd"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    obs = self._get_obs(data, state.info)
    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    return fall_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    actor_frame = self._get_actor_obs(data, info)

    info["rng"], obs_noise_rng = jax.random.split(info["rng"])
    noisy_state = self._apply_obs_noise(actor_frame, obs_noise_rng)

    actor_obs_history = jp.concatenate(
        [info["actor_obs_history"][1:], noisy_state[None, :]], axis=0
    )
    info["actor_obs_history"] = actor_obs_history

    privileged_state = self._get_critic_obs(data, info, actor_frame)

    return {
        "state": actor_obs_history.reshape(-1),
        "privileged_state": privileged_state,
    }

  def _apply_obs_noise(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
    """Apply PPO-style observation noise for actor inputs only."""
    frame_noise = jp.concatenate(
        [
            jp.ones(3) * 0.2,  # angular velocity
            jp.ones(3) * 0.05,  # projected gravity
            jp.zeros(3),  # commands
            jp.ones(12) * 0.01,  # joint positions
            jp.ones(12) * 0.01,  # joint velocities
            jp.zeros(12),  # actions
        ],
        axis=0,
    )
    return obs + (2.0 * jax.random.uniform(rng, obs.shape) - 1.0) * frame_noise

  def _get_actor_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    """Construct the actor observation vector."""
    return jp.concatenate(
        [
            self.get_local_angvel(data),
            self.get_gravity(data),
            info["command"],
            data.qpos[7:] - self._default_pose,
            data.qvel[6:],
            info["last_act"],
        ]
    )

  def _get_critic_obs(
      self, data: mjx.Data, info: dict[str, Any], actor_state: jax.Array
  ) -> jax.Array:
    """Construct the critic observation vector."""

    height_above_ground = data.qpos[2] - info["ground_z"]

    foot_height = data.site_xpos[self._feet_site_id, 2] - info["ground_z"]
    foot_contact = info["last_contact"].astype(jp.float32)
    foot_contact_forces = self._get_foot_contact_forces(data)

    privileged_state = jp.hstack([
        actor_state.ravel(),
        # Privileged information.
        self.get_local_linvel(data).ravel(),  # Linear velocity: 3.
        jp.array([height_above_ground]),
    ])
    return privileged_state
  
  def _get_foot_contact_forces(self, data: mjx.Data) -> jax.Array:
    """Returns per-foot ground contact force proxy from contact penetration."""
    contact_geoms = data.contact.geom
    foot_floor_mask = (
        (contact_geoms[:, 0][None, :] == self._feet_geom_id[:, None])
        & (contact_geoms[:, 1][None, :] == self._floor_geom_id)
    ) | (
        (contact_geoms[:, 1][None, :] == self._feet_geom_id[:, None])
        & (contact_geoms[:, 0][None, :] == self._floor_geom_id)
    )
    penetration = jp.maximum(-data.contact.dist, 0.0)
    return jp.sum(foot_floor_mask * penetration[None, :], axis=1)

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_local_angvel(data)
        ),
        "height": self._reward_base_height(data),
        "lin_vel_z": self._cost_lin_vel_z(self.get_local_linvel(data)),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "action": self._cost_action(action),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_local_angvel(data)),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "pose": self._reward_pose(data.qpos[7:]),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
    }

  # Tracking rewards.

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _reward_base_height(self, data: mjx.Data) -> jax.Array:
    # Reward base height.
    return jp.exp(-10.0 * jp.square(data.qpos[2] - self._config.reward_config.target_base_height))

  def _cost_lin_vel_z(self, local_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(local_linvel[2])

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  # Energy related rewards.
  
  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_action(self, act: jax.Array) -> jax.Array:
    return jp.sum(jp.square(act))

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))
  
  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  # Other rewards.

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    # Stay close to the default pose.
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

  # Perturbation and command sampling.

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    state.info["rng"], push_rng = jax.random.split(state.info["rng"])
    push_due = (state.info["step"] > 0) & (
        state.info["step"] % state.info["push_interval_steps"] == 0
    )
    velocity_push = jax.random.uniform(
        push_rng,
        (2,),
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )
    pushed_qvel = state.data.qvel.at[:2].set(velocity_push)
    xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
    data = state.data.replace(
        qvel=jp.where(push_due, pushed_qvel, state.data.qvel),
        xfrc_applied=xfrc_applied,
    )
    return state.replace(data=data)

  def _sample_ctrl_interval(self, rng: jax.Array) -> jax.Array:
    return jax.random.randint(
        rng,
        (),
        self._cmd_ctrl_interval[0],
        self._cmd_ctrl_interval[1] + 1,
    )

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng, val_rng, zero_rng = jax.random.split(rng, 3)
    raw = jax.random.uniform(
        val_rng, shape=(3,), minval=self._cmd_lo, maxval=self._cmd_hi
    )
    zero_mask = jax.random.uniform(zero_rng, (3,)) < self._cmd_zero_prob
    return jp.where(zero_mask, 0.0, raw)