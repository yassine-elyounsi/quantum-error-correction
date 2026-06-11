"""
A2C (Advantage Actor-Critic) + Probabilistic Policy Reuse (PPR)
agent for QEC encoding circuit discovery.

Architecture
------------
Two NumPy MLPs sharing nothing:

    actor   : obs_dim -> hidden -> hidden -> n_actions   (softmax)
    critic  : obs_dim -> hidden -> hidden -> 1           (value)

Update
------
On-policy, full-episode rollouts (configurable n-step) with GAE.

For each episode we collect (s_t, a_t, r_t, log_pi(a_t|s_t), V(s_t)).
After the episode:

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}     [GAE]
    R_t     = A_t + V(s_t)                                          [return target]

    actor_loss   = - mean_t  log_pi(a_t|s_t) * A_t
    critic_loss  = mean_t  (V(s_t) - R_t)^2
    entropy_loss = - mean_t  H(pi(.|s_t))

    total_loss   = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss

Probabilistic Policy Reuse
--------------------------
Same idea as before: with probability psi follow the library policy,
otherwise follow the actor. psi decays linearly during training.
Because A2C uses a stochastic policy, the library is also expected to
return a *sampled* action (not necessarily argmax). The agent's own
update only sees its own action choices (we mark library steps so
they don't contaminate the actor gradient).

Action masking
--------------
Before sampling, illegal-action logits are pushed to -inf so they
receive zero probability. The agent never picks an illegal action.

Noise-aware obs
---------------
The obs may contain an appended c_Z scalar (env.reward_mode='noise_aware').
The agent is agnostic to this -- it just reads obs_dim from the env.

Save / load
-----------
Weights are picklable np arrays. `agent.save(path)` and
`A2CAgent.load(path, env)` round-trip the policy.
"""

import os
import pickle
import numpy as np
from typing import Optional, Callable


# =====================================================================
# 1. NumPy MLP with manual backprop  (shared by actor and critic)
# =====================================================================

class MLP:
    """Two-hidden-layer ReLU MLP. Output activation is identity --
    the caller applies softmax (actor) or leaves linear (critic).
    """

    def __init__(self, in_dim, hidden, out_dim, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim),  size=(in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden),  size=(hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2 / hidden),  size=(hidden, out_dim)).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        self._cache = None

    def forward(self, x):
        # x: (B, in_dim)
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        out = a2 @ self.W3 + self.b3
        self._cache = (x, z1, a1, z2, a2, out)
        return out

    def backward(self, dout, max_grad_norm=None):
        """dout: gradient at the output. Returns nothing; updates weights."""
        x, z1, a1, z2, a2, _ = self._cache
        dW3 = a2.T @ dout
        db3 = dout.sum(axis=0)
        da2 = dout @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        if max_grad_norm is not None:
            total = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))
            if total > max_grad_norm and total > 0:
                scale = max_grad_norm / total
                for g in grads:
                    g *= scale

        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3

    # ---- serialization ----
    def state_dict(self):
        return {'W1': self.W1, 'b1': self.b1,
                'W2': self.W2, 'b2': self.b2,
                'W3': self.W3, 'b3': self.b3}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            setattr(self, k, v.astype(np.float32))


# =====================================================================
# 2. The A2C + PPR agent
# =====================================================================

def _softmax(logits):
    # numerically stable softmax over the last axis
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class A2CAgent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden=64,
        # ---- A2C hyperparameters ----
        lr=1e-3,
        gamma=0.95,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # ---- PPR ----
        library_policy: Optional[Callable] = None,
        psi_start=0.5, psi_end=0.05, psi_decay_steps=4000,
        # ---- misc ----
        seed=0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.actor  = MLP(obs_dim, hidden, n_actions, lr=lr, seed=seed)
        self.critic = MLP(obs_dim, hidden, 1,         lr=lr, seed=seed + 1)

        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm

        self.library_policy  = library_policy
        self.psi_start       = psi_start
        self.psi_end         = psi_end
        self.psi_decay_steps = psi_decay_steps

        self.rng = np.random.default_rng(seed)
        self.train_step = 0           # counts update() calls
        self.env_steps  = 0           # counts env.step() calls

    # ---- schedules ----
    def _psi(self):
        if self.library_policy is None:
            return 0.0
        frac = min(1.0, self.env_steps / max(1, self.psi_decay_steps))
        return self.psi_start + frac * (self.psi_end - self.psi_start)

    # ---- action selection ----------------------------------------
    def _policy_probs(self, obs, action_mask=None):
        """Forward the actor and return (probs, logits). Honours mask."""
        logits = self.actor.forward(obs[None, :])[0]
        if action_mask is not None:
            # set masked logits to a very negative number so e^logit ~ 0
            logits = np.where(action_mask, logits, -1e9)
        return _softmax(logits), logits

    def select_action(self, obs, action_mask=None, greedy=False):
        """
        Returns (action_idx, info_dict). info_dict contains:
            log_prob   : log pi(a | s) under the *agent's own* policy
            value      : V(s)
            from_library : whether the action came from the library
        For library actions, log_prob is still recorded as the agent's
        log-prob under its own policy (this is needed because PPR is
        an off-policy "hint" that we don't backprop through directly).
        """
        from_library = False

        # 1. PPR: with prob psi, ask the library
        if (self.library_policy is not None
                and not greedy
                and self.rng.random() < self._psi()):
            a = int(self.library_policy(obs))
            # if the library action is illegal under the mask, fall through
            # to the agent's own policy
            if action_mask is not None and not action_mask[a]:
                from_library = False
            else:
                from_library = True

        # 2. Agent's own policy
        probs, _ = self._policy_probs(obs, action_mask)
        if not from_library:
            if greedy:
                a = int(np.argmax(probs))
            else:
                a = int(self.rng.choice(self.n_actions, p=probs))

        # 3. Value estimate (needed only during training rollouts)
        v = float(self.critic.forward(obs[None, :])[0, 0])

        # log-prob of the *taken* action under the agent's own (masked) policy
        log_prob = float(np.log(probs[a] + 1e-12))

        return a, {'log_prob': log_prob, 'value': v, 'from_library': from_library,
                   'probs': probs}

    # ---- update from one rollout ---------------------------------
    def update(self, batch):
        """
        batch is a list of dicts, one per timestep, each containing:
            obs, action, reward, done, log_prob, value, from_library,
            action_mask  (optional)

        We compute GAE advantages and one gradient step on actor + critic.
        Returns a dict of scalar diagnostics.
        """
        T = len(batch)
        obs   = np.stack([b['obs']    for b in batch]).astype(np.float32)
        acts  = np.array([b['action'] for b in batch], dtype=np.int64)
        rews  = np.array([b['reward'] for b in batch], dtype=np.float32)
        dones = np.array([b['done']   for b in batch], dtype=np.float32)
        vals  = np.array([b['value']  for b in batch], dtype=np.float32)
        from_lib = np.array([b['from_library'] for b in batch], dtype=bool)

        # bootstrap value for the last state
        last_obs = batch[-1].get('next_obs')
        if last_obs is not None and not dones[-1]:
            last_v = float(self.critic.forward(last_obs[None, :].astype(np.float32))[0, 0])
        else:
            last_v = 0.0

        # --- GAE -----------------------------------------------------
        advs = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_v = vals[t + 1] if t + 1 < T else last_v
            delta = rews[t] + self.gamma * next_v * (1 - dones[t]) - vals[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advs[t] = gae
        returns = advs + vals
        # normalize advantages (a standard PPO/A2C trick)
        if advs.std() > 1e-8:
            advs_norm = (advs - advs.mean()) / advs.std()
        else:
            advs_norm = advs

        # --- forward both networks on the whole batch ----------------
        logits  = self.actor.forward(obs)                        # (T, A)
        # apply masks if present so log-probs match the on-policy probs
        masks = batch[0].get('action_mask')
        if masks is not None:
            mask_batch = np.stack([b.get('action_mask',
                                         np.ones(self.n_actions, dtype=bool))
                                   for b in batch])
            logits = np.where(mask_batch, logits, -1e9)
        probs   = _softmax(logits)                               # (T, A)
        log_probs_all = np.log(probs + 1e-12)
        chosen_logp = log_probs_all[np.arange(T), acts]          # (T,)

        # --- ACTOR gradient ------------------------------------------
        # We do NOT learn from library-supplied actions. They are
        # exploration hints, not part of the agent's own policy.
        weight = np.where(from_lib, 0.0, 1.0).astype(np.float32)

        # d/d(logits) of  -mean( advs * log pi(a|s) )
        #   for chosen action: -adv * (1 - p_a)
        #   for others       : +adv * p_a
        # We multiply by `weight` to zero out library steps.
        d_logits = probs.copy()
        d_logits[np.arange(T), acts] -= 1.0
        d_logits *= (advs_norm * weight)[:, None] / max(1, T)

        # --- entropy bonus -------------------------------------------
        # H = -sum p log p; we MAXIMISE entropy ↔ MINIMISE -H
        # dH/d(logits) = -p * (log p - sum_j p_j log p_j)
        entropy_per_step = -(probs * log_probs_all).sum(axis=-1)   # (T,)
        # gradient of mean entropy w.r.t logits:
        mean_logp = (probs * log_probs_all).sum(axis=-1, keepdims=True)  # (T,1)
        d_ent = -probs * (log_probs_all - mean_logp) / T
        d_logits_total = d_logits - self.ent_coef * d_ent

        self.actor.backward(d_logits_total, max_grad_norm=self.max_grad_norm)

        # --- CRITIC gradient -----------------------------------------
        v_pred = self.critic.forward(obs).flatten()              # (T,)
        d_v = (v_pred - returns)[:, None] * (2.0 * self.vf_coef / T)
        self.critic.backward(d_v, max_grad_norm=self.max_grad_norm)

        self.train_step += 1

        return {
            'actor_loss':   float(-((advs_norm * weight) * chosen_logp).mean()),
            'critic_loss':  float(((v_pred - returns) ** 2).mean()),
            'entropy':      float(entropy_per_step.mean()),
            'mean_return':  float(returns.mean()),
            'mean_advantage': float(advs.mean()),
            'n_library_steps': int(from_lib.sum()),
        }

    # ---- greedy policy export (for use as a PPR library later) ---
    def greedy_policy(self):
        """Return a callable obs -> int that picks argmax of the actor."""
        def pi(obs):
            probs, _ = self._policy_probs(obs)
            return int(np.argmax(probs))
        return pi

    def stochastic_policy(self):
        """Return a callable obs -> int that samples from the actor."""
        def pi(obs):
            probs, _ = self._policy_probs(obs)
            return int(self.rng.choice(self.n_actions, p=probs))
        return pi

    # ---- save / load ---------------------------------------------
    def save(self, path):
        payload = {
            'obs_dim': self.obs_dim,
            'n_actions': self.n_actions,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'train_step': self.train_step,
            'env_steps': self.env_steps,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path, **kwargs):
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        agent = cls(obs_dim=payload['obs_dim'],
                    n_actions=payload['n_actions'],
                    **kwargs)
        agent.actor.load_state_dict(payload['actor'])
        agent.critic.load_state_dict(payload['critic'])
        agent.train_step = payload.get('train_step', 0)
        agent.env_steps  = payload.get('env_steps', 0)
        return agent


# =====================================================================
# 3. Rollout collector  (single-env, one full episode)
# =====================================================================

def collect_episode(env, agent, max_steps=None, greedy=False):
    """
    Roll the agent through one episode and return:
        batch (list of transition dicts), episode_info (dict).
    """
    if max_steps is None:
        max_steps = env.max_steps
    obs = env.reset()
    batch = []
    total_reward = 0.0
    success = False

    for t in range(max_steps):
        agent.env_steps += 1
        mask = env.action_mask() if hasattr(env, 'action_mask') else None
        action, info = agent.select_action(obs, action_mask=mask, greedy=greedy)
        next_obs, reward, done, env_info = env.step(action)
        batch.append({
            'obs': obs.copy(),
            'next_obs': next_obs.copy(),
            'action': action,
            'reward': reward,
            'done': float(done),
            'log_prob': info['log_prob'],
            'value': info['value'],
            'from_library': info['from_library'],
            'action_mask': mask if mask is not None else np.ones(agent.n_actions, dtype=bool),
        })
        total_reward += reward
        obs = next_obs
        if done:
            success = bool(env_info.get('success', False))
            break

    episode_info = {
        'total_reward': total_reward,
        'success': success,
        'length': len(batch),
        'history': list(env.history) if hasattr(env, 'history') else [],
    }
    return batch, episode_info


# =====================================================================
# 4. Multi-agent runner
# =====================================================================

def train_multi_agent(
    make_env: Callable,
    make_agent: Callable,
    num_agents: int,
    num_episodes: int,
    log_every: int = 100,
    seeds: Optional[list] = None,
    verbose: bool = True,
):
    """
    Train `num_agents` independent A2C agents in parallel (sequentially in
    Python, but with independent state and seeds -- a NumPy-faithful
    analogue of the paper's "many parallel agents on one GPU" approach).

    Each agent uses its own env and its own random seed. The function
    returns the best agent (by shortest successful encoding) and a
    summary of all runs.

    Args:
        make_env   : callable () -> env
        make_agent : callable (env, seed) -> A2CAgent
        num_agents : how many independent runs
        num_episodes : episodes per run
        log_every  : print progress every N episodes
        seeds      : optional list of length num_agents; else 0..num_agents-1
        verbose    : print progress

    Returns:
        dict with keys:
            'best_agent', 'best_env',
            'best_circuit', 'best_circuit_len',
            'agents', 'envs', 'histories'
    """
    if seeds is None:
        seeds = list(range(num_agents))
    assert len(seeds) == num_agents

    envs   = [make_env()                    for _ in range(num_agents)]
    agents = [make_agent(envs[i], seeds[i]) for i in range(num_agents)]
    histories = [[] for _ in range(num_agents)]
    best_per_agent = [{'len': float('inf'), 'circuit': None} for _ in range(num_agents)]

    for ep in range(num_episodes):
        for ai in range(num_agents):
            batch, info = collect_episode(envs[ai], agents[ai])
            agents[ai].update(batch)
            histories[ai].append(info)
            if info['success'] and info['length'] < best_per_agent[ai]['len']:
                best_per_agent[ai] = {'len': info['length'],
                                      'circuit': info['history']}
        if verbose and (ep + 1) % log_every == 0:
            line = f"  ep {ep+1:5d}"
            for ai in range(num_agents):
                last = histories[ai][-log_every:]
                sr = np.mean([h['success'] for h in last])
                line += f"  | A{ai}: SR={sr:.2f} best={best_per_agent[ai]['len']}"
            print(line)

    # pick the overall best
    ranked = sorted(range(num_agents),
                    key=lambda i: best_per_agent[i]['len'])
    best_i = ranked[0]
    return {
        'best_agent_idx': best_i,
        'best_agent': agents[best_i],
        'best_env':   envs[best_i],
        'best_circuit': best_per_agent[best_i]['circuit'],
        'best_circuit_len': best_per_agent[best_i]['len'],
        'best_per_agent': best_per_agent,
        'agents': agents,
        'envs':   envs,
        'histories': histories,
    }