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
"""Domain randomization for the Go2 environment."""

import jax
import jax.numpy as jp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1

FRICTION_RANGE = (0.5, 2.0)
MASS_RANGE = (0.85, 1.15)
KP_RANGE = (25.0, 45.0)
KD_RANGE = (0.3, 1.6)
COM_OFFSET_RANGE = (0.05, 0.05, 0.04)


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # JAVE-style per-episode friction scale: *U(0.5, 2.0).
    rng, key = jax.random.split(rng)
    friction_scale = jax.random.uniform(
        key, minval=FRICTION_RANGE[0], maxval=FRICTION_RANGE[1]
    )
    geom_friction = model.geom_friction * friction_scale

    # JAVE-style mass scale: body_mass and body_inertia *U(0.85, 1.15).
    rng, key = jax.random.split(rng)
    mass_scale = jax.random.uniform(
        key, minval=MASS_RANGE[0], maxval=MASS_RANGE[1]
    )
    body_mass = model.body_mass * mass_scale
    body_inertia = model.body_inertia * mass_scale

    # JAVE-style per-episode absolute actuator gains.
    rng, key = jax.random.split(rng)
    kp = jax.random.uniform(key, minval=KP_RANGE[0], maxval=KP_RANGE[1])
    rng, key = jax.random.split(rng)
    kd = jax.random.uniform(key, minval=KD_RANGE[0], maxval=KD_RANGE[1])
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)
    actuator_biasprm = actuator_biasprm.at[:, 2].set(-kd)

    # JAVE-style torso COM offset.
    rng, key = jax.random.split(rng)
    com_offset = jax.random.uniform(
        key,
        (3,),
        minval=-jp.array(COM_OFFSET_RANGE),
        maxval=jp.array(COM_OFFSET_RANGE),
    )
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + com_offset
    )

    return (
        geom_friction,
        body_ipos,
        body_mass,
        body_inertia,
        actuator_gainprm,
        actuator_biasprm,
    )

  (
      friction,
      body_ipos,
      body_mass,
      body_inertia,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "body_inertia": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "body_inertia": body_inertia,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  return model, in_axes
