import jax
import jax.numpy as jnp
from gxm import spaces 
from gxm import Environment, Timestep
from dataclasses import asdict, dataclass

class GxmWrapper:
    # def __init__(self, env: Environment, n_envs, key):
    def __init__(self, env: Environment, key):
        self.env = env
        self.num_envs = key.shape[0] 
        self.single_action_space = env.action_space

        # self.single_observation_space = self.env.gymnax_to_gxm_space(
        #     self.env.env.observation_space(self.env.env_params)
        # )
        self.single_observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(64, 64, 1),
        )
        self.action_space = self.single_action_space
        self.observation_space = spaces.Box(
            low=self.single_observation_space.low,
            high=self.single_observation_space.high,
            shape=(self.num_envs,) + self.single_observation_space.shape,
        )
        self.env.resize = self.resize

    def reset(self, key):
        env_state, timestep = jax.vmap(self.env.init)(key)
        return env_state, timestep

    def step(self, actions, env_state, key):
        env_state, timestep = jax.vmap(self.env.step, in_axes=(None, 0, 0))(key, env_state, actions)
        return env_state, timestep

    def resize(cls, obs: jax.Array) -> jax.Array:
        """Resize an observation to 64x64 (Dreamer)."""
        out = jax.image.resize(obs, (64, 64, 1), method="bilinear")
        return out