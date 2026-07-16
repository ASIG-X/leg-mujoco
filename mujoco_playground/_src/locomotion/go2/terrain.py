"""Implicit terrain perturbations for Go2 training."""

import jax
import jax.numpy as jp


def sample_fourier_terrain(
    rng,
    *,
    difficulty,
    n_fourier: int,
    freq_range: tuple,
    slope_max_deg: float,
    flat_prob: float,
):
    """Sample deterministic Fourier heightfield parameters for one episode."""
    k_freq_mag, k_freq_dir, k_phase, k_flat = jax.random.split(rng, 4)

    freq_mag = jax.random.uniform(
        k_freq_mag,
        (n_fourier,),
        minval=freq_range[0],
        maxval=freq_range[1],
    )
    freq_dir = jax.random.uniform(
        k_freq_dir,
        (n_fourier,),
        minval=0.0,
        maxval=2 * jp.pi,
    )
    terrain_freqs = freq_mag[:, None] * jp.stack(
        [jp.cos(freq_dir), jp.sin(freq_dir)],
        axis=-1,
    )
    terrain_phases = jax.random.uniform(
        k_phase,
        (n_fourier,),
        minval=0.0,
        maxval=2 * jp.pi,
    )

    slope_angle_max = slope_max_deg * jp.pi / 180.0
    target_grad_rms = jp.tan(slope_angle_max * difficulty)
    terrain_amps = target_grad_rms / (freq_mag * jp.sqrt(n_fourier / 2.0) + 1e-8)

    is_flat = jax.random.uniform(k_flat) < flat_prob
    terrain_amps = jp.where(is_flat, jp.zeros(n_fourier), terrain_amps)
    return terrain_freqs, terrain_phases, terrain_amps


def fourier_terrain_gradient(xy, terrain_freqs, terrain_phases, terrain_amps):
    """Evaluate the local heightfield gradient."""
    phase_args = jp.sum(terrain_freqs * xy[None, :], axis=-1) + terrain_phases
    return jp.sum(
        terrain_amps[:, None] * terrain_freqs * jp.cos(phase_args)[:, None],
        axis=0,
    )


def fourier_terrain_force(xy, terrain_freqs, terrain_phases, terrain_amps, weight):
    """Convert local Fourier slope into an equivalent body force."""
    grad = fourier_terrain_gradient(xy, terrain_freqs, terrain_phases, terrain_amps)
    return weight * jp.array([grad[0], grad[1], 0.0])


def differentiated_ou_foot_forces(
    foot_bump_ou,
    innovations,
    normal_forces,
    *,
    difficulty,
    std,
    decay,
    robot_weight,
):
    """Update dOU foot-bump state and return contact-scaled forces."""
    effective_std = std * difficulty
    next_ou = (1.0 - decay) * foot_bump_ou + effective_std * innovations.astype(
        jp.float64
    )
    delta = next_ou - foot_bump_ou
    force_scale = normal_forces / jp.maximum(robot_weight, 1e-6)
    return next_ou, delta * force_scale[:, None]