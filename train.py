import argparse
import dataclasses
from typing import Dict, Sequence
import wandb

import numpy as np

from dreamerv3_flax.buffer import ReplayBuffer
from dreamerv3_flax.jax_agent import JAXAgent
import gxm
from dreamerv3_flax.wrapper import GxmWrapper
import jax
import jax.numpy as jnp
import flashbax as fbx


def get_eval_metric(achievements: Sequence[Dict]) -> float:
    """For crafter env only."""
    achievements = [list(achievement.values()) for achievement in achievements]
    success_rate = 100 * (np.array(achievements) > 0).mean(axis=0)
    score = np.exp(np.mean(np.log(1 + success_rate))) - 1
    eval_metric = {
        "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
        "score": score,
    }
    return eval_metric

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Timestep(gxm.Timestep):
    first: jnp.ndarray
    action: jnp.ndarray

def main(args):
    # Logger
    project = "dreamerv3-flax"
    group = f"{args.exp_name}"
    if args.timestamp:
        group += f"-{args.timestamp}"
    name = f"s{args.seed}"
    logger = wandb.init(project=project, group=group, name=name)

    # Seed
    np.random.seed(args.seed)

    # Setup stepping etc.
    steps_per_epoch = args.n_steps * args.n_envs #number of total environment interactions per epoch
    epochs = args.total_steps // steps_per_epoch #total number of epochs

    # Environment
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, args.n_envs)
    atari_wrapper_kwargs = dict(frame_stack_size=1)
    #Note: Expects Env to be jax-compatible and auto-resetting
    env = GxmWrapper(gxm.make("JAXAtari/pong", atari_wrapper_kwargs=atari_wrapper_kwargs), key=keys)
    # env_state, timestep = env.init(key=key)

    # Buffer
    # orig_buffer = ReplayBuffer(env, batch_size=16, num_steps=64, buffer_size=5*10**6) #5M as in original dreamerv3
    buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=args.n_steps, # Maximum length of the buffer along the time axis.
        min_length_time_axis=64, # Minimum length across the time axis before we can sample.
        sample_batch_size=16, # Batch size of trajectories sampled from the buffer.
        add_batch_size=args.n_envs, # Batch size of trajectories added to the buffer.
        sample_sequence_length=64, # Sequence length of trajectories sampled from the buffer.
        period=1 # Period at which we sample trajectories from the buffer.
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    # Agent
    agent = JAXAgent(env, seed=args.seed)
    agent_state = agent.initial_state(args.n_envs)

    # Reset
    # actions = jax.vmap(env.action_space.sample)(key=keys)
    env_state, timestep = env.reset(keys)
    dones = jnp.ones_like(timestep.truncated).astype(jnp.bool)
    actions = agent.act(timestep.obs, dones, agent_state)[0]
    action = jnp.argmax(actions, axis=-1).astype(jnp.int32)
    env_state, ts_env = env.step(action, env_state, key)
    timestep = Timestep(**dataclasses.asdict(ts_env), first=dones, action=actions)
    init_timestep = jax.tree.map(lambda x: x[0], timestep) #get first env only 
    buffer_state = buffer.init(init_timestep)

    def scanned_step(carry, x):
        env_state, agent_state, actions, first, key = carry
        action = jnp.argmax(actions, axis=-1).astype(jnp.int32)
        env_state, ts_env = env.step(action, env_state, key)
        timestep = Timestep(**dataclasses.asdict(ts_env), first=first, action=actions)
        # actions, agent_state = agent.act(timestep.obs, first, agent_state)
        actions, agent_state, key = agent.key_act(key, timestep.obs, first, agent_state)
        _, key = jax.random.split(key)
        return (env_state, agent_state, actions, timestep.done, key), (timestep, actions)

    # Train
    for epoch in range(epochs):
        _, key = jax.random.split(key)
        if epoch != 0:
            actions = actions[-1] # select from last step
        final_vals, (timestep, actions) = jax.lax.scan(
            scanned_step,
            (env_state, agent_state, actions, dones, key),
            None,
            length=args.n_steps,
        )
        env_state = final_vals[0]
        # obs shape: (n_steps, n_envs, 64, 64, 1)
        # but buffer expectes (n_envs, n_steps, 64, 64, 1)
        timestep_buf = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), timestep)
        # print shapes of timestep_buf
        buffer_state = buffer.add(buffer_state, timestep_buf)
        # for env in range(args.n_envs):
        #     buffer.add(timestep.obs[:, env], actions[:, env], timestep.reward[:, env], jnp.logical_or(timestep.truncated[:, env], timestep.terminated[:, env]), first[:, env])
        # for step in range(args.n_steps): 
        #     orig_buffer.add(timestep.obs[step], actions[step], timestep.reward[step], jnp.logical_or(timestep.truncated[step], timestep.terminated[step]), timestep.terminated[step])

        returns = timestep.info["returned_episode_returns"].mean()
        lengths = timestep.info["returned_episode_lengths"].mean()
        done_infos = {"episode_returns": returns, "episode_lengths": lengths}
        print(f"Step: {epoch}, done_infos: {done_infos}") 
        # logger.log(done_infos, epoch)

        # if epoch >= 1024 and epoch % 2 == 0:
        if epoch * steps_per_epoch < 10_000:
            continue
        exp = buffer.sample(buffer_state, key).experience
        # orig_data = orig_buffer.sample()
        data = {
            'obs': exp.obs,
            'action': exp.action,
            'reward': exp.reward,
            'cont': 1.0 - exp.done,
            'first': exp.first
        }

        _, train_metric = agent.train(data)
        # if epoch % 100 == 0:
            # remove all _hist from train_metric, since it leads to bugs with wandb
        train_metric = {k: v for k, v in train_metric.items() if "hist" not in k}
        train_metric.update(done_infos)
        logger.log(train_metric, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_steps", default=128, type=int) #n_steps for each grad update
    parser.add_argument("--n_envs", default=128, type=int) #16 is original dreamerv3 
    # parser.add_argument("--total_steps", default=200_000_000, type=int) #Frame budget
    parser.add_argument("--total_steps", default=10_000_000_000, type=int) #Frame budget
    args = parser.parse_args()
    #TODO: 
    # - Check results
    # - Move logging to callback(?) 
    # - Log decoded imaginations
    # - Add rtpt
    # - Checkpointing

    main(args)
