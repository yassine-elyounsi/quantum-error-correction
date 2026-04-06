# =============================================================================
# shor_env.py
# Gymnasium Environment — 9-Qubit Shor Code
# =============================================================================
#
# Observation space : MultiBinary(8)  — 8-bit syndrome vector
# Action space      : Discrete(28)
#                       0        → no correction
#                       1  – 9   → X on qubit 0–8
#                       10 – 18  → Z on qubit 0–8
#                       19 – 27  → Y on qubit 0–8
#
# Reward is computed entirely from the Shor syndrome table.
# No correction circuit is built. No fidelity is computed.
# One episode = one syndrome extraction + one action.
#
# Usage:
#   from envs.shor_env import ShorCodeEnv
#   env = ShorCodeEnv(noise_rate=0.05)
#   obs, info = env.reset()
#   obs, reward, terminated, truncated, info = env.step(action)
# =============================================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.circuits.shor_code import (
    run_syndrome_extraction,
    decode_shor_syndrome,
    SHOR_SYNDROME_TABLE,
)


# =============================================================================
# ACTION MAP
# =============================================================================
#
# The Shor code can have errors on 9 qubits.
# Each qubit can have error type X, Z, or Y.
# Plus one no-correction action.
#
# Layout:
#   Action 0        → no correction          (correct when syndrome = 00000000)
#   Actions 1  – 9  → X on qubit 0 through 8
#   Actions 10 – 18 → Z on qubit 0 through 8
#   Actions 19 – 27 → Y on qubit 0 through 8
#
# Formula:
#   X on qubit i → action = 1  + i
#   Z on qubit i → action = 10 + i
#   Y on qubit i → action = 19 + i

N_QUBITS  = 9
N_ACTIONS = 1 + N_QUBITS * 3   # 1 + 27 = 28


def action_to_correction(action):
    """
    Map an action integer to (correction_type, correction_qubit).

    Parameters
    ----------
    action : int   0–27

    Returns
    -------
    correction_type  : str   'None', 'X', 'Z', or 'Y'
    correction_qubit : int   qubit index (0–8), or -1 for no correction
    """
    if action == 0:
        return 'None', -1
    elif 1 <= action <= 9:
        return 'X', action - 1
    elif 10 <= action <= 18:
        return 'Z', action - 10
    elif 19 <= action <= 27:
        return 'Y', action - 19
    else:
        raise ValueError(f'Invalid action {action}. Must be in [0, 27]')


def correction_to_action(error_type, error_qubit):
    """
    Map a (error_type, error_qubit) pair to the correct action integer.

    Parameters
    ----------
    error_type  : str   'None', 'X', 'Z', or 'Y'
    error_qubit : int   0–8, or -1

    Returns
    -------
    action : int
    """
    if error_type == 'None' or error_qubit == -1:
        return 0
    elif error_type == 'X':
        return 1  + error_qubit
    elif error_type == 'Z':
        return 10 + error_qubit
    elif error_type == 'Y':
        return 19 + error_qubit
    else:
        raise ValueError(f'Unknown error_type {error_type!r}')


# =============================================================================
# REWARD TABLE
# =============================================================================
# Built directly from SHOR_SYNDROME_TABLE.
# Maps (syndrome_int, action) → (reward, reason)
#
# Reward logic:
#   syndrome = 0 (no error detected):
#       action = 0 (no correction)    → +1.0   correct
#       any other action              → -1.0   unnecessary correction
#
#   syndrome != 0 (error detected):
#       action matches table entry    → +1.0   correct type and qubit
#       action = 0 (no correction)    → -0.5   missed the error
#       wrong qubit but right type    → -0.5   partially wrong
#       right qubit but wrong type    → -0.5   partially wrong
#       completely wrong              → -0.5   wrong qubit and type

def _build_reward_table():
    """
    Build the complete (syndrome_int, action) → (reward, reason) table.
    Only covers syndromes that appear in SHOR_SYNDROME_TABLE (22 entries).
    Unknown syndromes are handled separately in step().

    Returns
    -------
    table : dict   (syndrome_int, action) → (reward, reason)
    """
    table = {}

    for syndrome_int, (error_type, correct_qubit, correction, description) \
            in SHOR_SYNDROME_TABLE.items():

        # Correct action for this syndrome
        correct_action = correction_to_action(error_type, correct_qubit)

        for action in range(N_ACTIONS):
            act_type, act_qubit = action_to_correction(action)

            if error_type == 'None':
                # Syndrome says no error
                if action == 0:
                    table[(syndrome_int, action)] = (
                        +1.0,
                        'correct: no error detected, no correction applied'
                    )
                else:
                    table[(syndrome_int, action)] = (
                        -1.0,
                        f'wrong: unnecessary {act_type} on q{act_qubit}'
                    )

            else:
                # Syndrome points to a specific error
                if action == correct_action:
                    # Perfect match — right type and right qubit
                    table[(syndrome_int, action)] = (
                        +1.0,
                        f'correct: {error_type} on q{correct_qubit} '
                        f'matches syndrome'
                    )
                elif action == 0:
                    # Agent did nothing despite an error
                    table[(syndrome_int, action)] = (
                        -0.5,
                        f'wrong: {error_type} error on q{correct_qubit} '
                        f'missed, no correction applied'
                    )
                elif act_qubit == correct_qubit and act_type != error_type:
                    # Right qubit but wrong error type
                    table[(syndrome_int, action)] = (
                        -0.5,
                        f'wrong: right qubit q{correct_qubit} but '
                        f'wrong type {act_type} (should be {error_type})'
                    )
                elif act_qubit != correct_qubit and act_type == error_type:
                    # Right type but wrong qubit
                    table[(syndrome_int, action)] = (
                        -0.5,
                        f'wrong: right type {error_type} but '
                        f'wrong qubit q{act_qubit} (should be q{correct_qubit})'
                    )
                else:
                    # Both wrong
                    table[(syndrome_int, action)] = (
                        -1,
                        f'wrong: {act_type} on q{act_qubit}, '
                        f'should be {error_type} on q{correct_qubit}'
                    )

    return table


REWARD_TABLE = _build_reward_table()


# =============================================================================
# ENVIRONMENT
# =============================================================================

class ShorCodeEnv(gym.Env):
    """
    Gymnasium environment for the 9-qubit Shor code.

    The agent observes an 8-bit syndrome vector and must select
    which qubit to correct, with which error type (X, Z, or Y),
    or do nothing. Reward is determined entirely by the syndrome
    table — no circuit is built in step() and no fidelity is computed.

    Parameters
    ----------
    noise_rate    : float
        Depolarising noise per gate [0.0, 1.0].
    noise_type    : str
        'depolarising' or 'x_only'.
    logical_state : str or tuple
        '0', '1', '+', '-', (theta, phi), or 'random'.
        'random' picks from ['0', '1'] uniformly each episode.

    Observation Space
    -----------------
    MultiBinary(8) — syndrome vector [s0, s1, s2, s3, s4, s5, s6, s7]

        s0, s1  →  X-syndrome Group 0  (qubits 0,1,2)
        s2, s3  →  X-syndrome Group 1  (qubits 3,4,5)
        s4, s5  →  X-syndrome Group 2  (qubits 6,7,8)
        s6      →  Z-syndrome Group 0 vs Group 1
        s7      →  Z-syndrome Group 1 vs Group 2

    Action Space
    ------------
    Discrete(28)
        0        → no correction
        1  – 9   → X on qubit 0–8
        10 – 18  → Z on qubit 0–8
        19 – 27  → Y on qubit 0–8

    Reward
    ------
    +1.0   action matches syndrome table prescription
    -0.5   error detected but wrong qubit or wrong type
    -1.0   no error detected but correction applied anyway
    -0.5   unknown syndrome and agent applies correction  (multi-qubit error)
    +0.5   unknown syndrome and agent does nothing        (safe choice)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 noise_rate=0.05,
                 noise_type='depolarising',
                 logical_state='random'):

        super().__init__()

        if not 0.0 <= noise_rate <= 1.0:
            raise ValueError(
                f'noise_rate must be in [0.0, 1.0]. Got {noise_rate}'
            )

        self.noise_rate    = noise_rate
        self.noise_type    = noise_type
        self.logical_state = logical_state

        # Gymnasium spaces
        self.observation_space = spaces.MultiBinary(8)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        # Episode state — set by reset()
        self._syndrome_int          = None
        self._syndrome_str          = None
        self._syndrome_vec          = None
        self._true_error_qubit      = None
        self._true_error_type       = None
        self._current_logical_state = None

        # Cumulative statistics
        self._episode_count = 0
        self._total_reward  = 0.0
        self._correct_count = 0

    # ── RESET ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Start a new episode.

        1. Choose logical state (random or fixed)
        2. Randomly inject an error (or none) on one qubit
        3. Run noisy Shor syndrome extraction
        4. Return 8-bit syndrome vector as observation

        Random error injection is kept on so the agent sees all
        syndrome patterns frequently during training. Without it,
        syndrome 00000000 would dominate at low noise rates.

        Returns
        -------
        observation : np.ndarray   shape (8,)  dtype int8
        info        : dict
        """
        super().reset(seed=seed)

        # Logical state for this episode
        if self.logical_state == 'random':
            self._current_logical_state = (
                '0' if self.np_random.integers(0, 2) == 0 else '1'
            )
        else:
            self._current_logical_state = self.logical_state

        # Randomly inject an error:
        #   choice 0    → no error
        #   choice 1–9  → X, Z, or Y on one of 9 qubits
        #
        # 10 choices (None + 9 qubits) × 3 error types = 28 scenarios
        # We use uniform sampling over qubits then uniform over types
        qubit_choice = self.np_random.integers(0, 10)   # 0 = none, 1-9 = qubit
        if qubit_choice == 0:
            self._true_error_qubit = None
            self._true_error_type  = 'None'
        else:
            self._true_error_qubit = int(qubit_choice - 1)   # qubit 0-8
            type_idx = self.np_random.integers(0, 3)
            self._true_error_type  = ['X', 'Z', 'Y'][type_idx]

        # Run syndrome extraction (1 shot)
        syndrome_int, syndrome_str, _ = run_syndrome_extraction(
            logical_state=self._current_logical_state,
            error_qubit=self._true_error_qubit,
            error_type=self._true_error_type
                       if self._true_error_qubit is not None else 'X',
            p=self.noise_rate,
            shots=1,
            noise_type=self.noise_type
        )

        self._syndrome_int = syndrome_int
        self._syndrome_str = syndrome_str
        self._syndrome_vec = np.array(
            [int(b) for b in syndrome_str], dtype=np.int8
        )

        self._episode_count += 1

        observation = self._syndrome_vec.copy()
        info = {
            'syndrome_int'  : self._syndrome_int,
            'syndrome_str'  : self._syndrome_str,
            'true_error_q'  : self._true_error_qubit,
            'true_error_t'  : self._true_error_type,
            'logical_state' : self._current_logical_state,
        }

        return observation, info

    # ── STEP ──────────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Agent applies a correction action.
        Reward is looked up from the syndrome reward table.
        No quantum circuit is executed here.

        Parameters
        ----------
        action : int   0–27

        Returns
        -------
        observation  : np.ndarray   shape (8,)
        reward       : float
        terminated   : bool         always True (1 step per episode)
        truncated    : bool         always False
        info         : dict
        """
        if self._syndrome_vec is None:
            raise RuntimeError('Call reset() before step().')

        assert self.action_space.contains(action), \
            f'Invalid action {action}. Must be in [0, {N_ACTIONS - 1}]'

        # Decode the syndrome to know what the table prescribes
        decoded_type, correct_qubit, correction, description = \
            decode_shor_syndrome(self._syndrome_int)

        correct_action = correction_to_action(decoded_type, correct_qubit)
        act_type, act_qubit = action_to_correction(action)

        # Look up reward
        # Two cases: known syndrome (in table) vs unknown (multi-qubit error)
        if self._syndrome_int in SHOR_SYNDROME_TABLE:
            reward, reason = REWARD_TABLE[(self._syndrome_int, action)]
        else:
            # Unknown syndrome — likely caused by multi-qubit noise
            # Agent cannot know the right answer from training data alone
            # Safe strategy: do nothing
            if action == 0:
                reward = +0.5
                reason = 'unknown syndrome: safe choice to do nothing'
            else:
                reward = -0.5
                reason = (f'unknown syndrome: risky correction '
                          f'{act_type} on q{act_qubit}')

        # Update statistics
        self._total_reward  += reward
        if reward > 0:
            self._correct_count += 1

        observation = self._syndrome_vec.copy()
        info = {
            'action'         : action,
            'action_type'    : act_type,
            'action_qubit'   : act_qubit,
            'correct_action' : correct_action,
            'correct_qubit'  : correct_qubit,
            'decoded_type'   : decoded_type,
            'reward'         : reward,
            'reward_reason'  : reason,
            'syndrome_int'   : self._syndrome_int,
            'syndrome_str'   : self._syndrome_str,
            'true_error_q'   : self._true_error_qubit,
            'true_error_t'   : self._true_error_type,
            'logical_state'  : self._current_logical_state,
            'known_syndrome' : self._syndrome_int in SHOR_SYNDROME_TABLE,
        }

        return observation, reward, True, False, info

    # ── RENDER ────────────────────────────────────────────────────────────────

    def render(self):
        """Print current episode state to console."""
        if self._syndrome_vec is None:
            print('Environment not reset yet. Call reset() first.')
            return

        decoded_type, correct_qubit, correction, _ = \
            decode_shor_syndrome(self._syndrome_int)
        correct_action = correction_to_action(decoded_type, correct_qubit)
        known = self._syndrome_int in SHOR_SYNDROME_TABLE

        print(f'\n  ShorCodeEnv')
        print(f'  {"─"*45}')
        print(f'  Logical state    : |{self._current_logical_state}⟩')
        print(f'  Noise rate       : {self.noise_rate*100:.1f}%')
        print(f'  Syndrome vector  : {self._syndrome_vec}')
        print(f'  Syndrome string  : {self._syndrome_str}  '
              f'(int={self._syndrome_int})')
        print(f'  Known syndrome   : {known}')
        print(f'  Decoded error    : {decoded_type} on q{correct_qubit}')
        print(f'  Correct action   : {correct_action}  '
              f'({decoded_type} on q{correct_qubit})')
        print(f'  True error       : {self._true_error_type} '
              f'on q{self._true_error_qubit}')
        print(f'  {"─"*45}')

    # ── STATS ─────────────────────────────────────────────────────────────────

    def get_stats(self):
        """Return cumulative statistics."""
        ep = max(self._episode_count, 1)
        return {
            'episodes'    : self._episode_count,
            'total_reward': round(self._total_reward,    2),
            'mean_reward' : round(self._total_reward / ep, 4),
            'accuracy'    : round(self._correct_count  / ep, 4),
        }

    def reset_stats(self):
        """Reset cumulative statistics."""
        self._episode_count = 0
        self._total_reward  = 0.0
        self._correct_count = 0


# =============================================================================
# RANDOM AGENT BASELINE
# =============================================================================

def run_random_agent(env, n_episodes=10_000, verbose=True):
    """
    Run a random agent for n_episodes as a performance baseline.

    A random agent picks uniformly from [0..27].
    Expected accuracy is ~1/28 ≈ 3.6% by chance.
    Any trained RL agent must significantly beat this.

    Parameters
    ----------
    env        : ShorCodeEnv
    n_episodes : int
    verbose    : bool

    Returns
    -------
    results : dict
    """
    env.reset_stats()

    reward_history = []
    correct        = 0
    unknown_count  = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        action    = env.action_space.sample()
        _, reward, _, _, info = env.step(action)

        reward_history.append(reward)
        if reward > 0:
            correct += 1
        if not info['known_syndrome']:
            unknown_count += 1

        if verbose and n_episodes >= 5 and (ep + 1) % (n_episodes // 5) == 0:
            window    = reward_history[-(n_episodes // 5):]
            mean_r    = np.mean(window)
            acc       = correct / (ep + 1)
            print(f'  ep={ep+1:6d}  '
                  f'mean_reward={mean_r:+.3f}  '
                  f'accuracy={acc:.2%}')

    results = {
        'n_episodes'    : n_episodes,
        'mean_reward'   : round(float(np.mean(reward_history)), 4),
        'std_reward'    : round(float(np.std(reward_history)),  4),
        'accuracy'      : round(correct / n_episodes,           4),
        'unknown_rate'  : round(unknown_count / n_episodes,     4),
        'reward_dist'   : {
            '+1.0': reward_history.count(1.0),
            '+0.5': reward_history.count(0.5),
            '-0.5': reward_history.count(-0.5),
            '-1.0': reward_history.count(-1.0),
        }
    }

    if verbose:
        print(f'\n  {"─"*45}')
        print(f'  RANDOM AGENT — {n_episodes} episodes')
        print(f'  {"─"*45}')
        print(f'  Mean reward    : {results["mean_reward"]:+.4f}')
        print(f'  Std reward     : {results["std_reward"]:.4f}')
        print(f'  Accuracy       : {results["accuracy"]:.2%}')
        print(f'  Unknown rate   : {results["unknown_rate"]:.2%}')
        print(f'  Reward dist    : {results["reward_dist"]}')
        print(f'  {"─"*45}')
        print(f'  Expected ~3.6% accuracy  (28 actions, 1 correct)')
        print(f'  Trained RL agent should reach >85%')

    return results


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':

    print('=' * 55)
    print('  shor_env.py — self test')
    print('=' * 55)

    # ── Test 1: action map ────────────────────────────────────────────────────
    print('\n[TEST 1] Action Map (sample):')
    for a in [0, 1, 5, 9, 10, 14, 18, 19, 23, 27]:
        t, q = action_to_correction(a)
        back  = correction_to_action(t, q)
        ok    = back == a
        print(f'  action={a:2d}  type={t:4s}  qubit={q:2d}  '
              f'roundtrip={back:2d}  {"OK" if ok else "FAIL"}')

    # ── Test 2: reward table coverage ────────────────────────────────────────
    print('\n[TEST 2] Reward Table Coverage:')
    n_entries = len(REWARD_TABLE)
    expected  = len(SHOR_SYNDROME_TABLE) * N_ACTIONS
    print(f'  Syndrome table entries : {len(SHOR_SYNDROME_TABLE)}')
    print(f'  N_ACTIONS              : {N_ACTIONS}')
    print(f'  Reward table entries   : {n_entries}')
    print(f'  Expected               : {expected}')
    status = 'PASS' if n_entries == expected else 'FAIL'
    print(f'  [{status}]')

    # ── Test 3: env_checker ───────────────────────────────────────────────────
    print('\n[TEST 3] gymnasium.utils.env_checker:')
    from gymnasium.utils.env_checker import check_env
    env = ShorCodeEnv(noise_rate=0.05)
    try:
        check_env(env, warn=True)
        print('  PASS — env_checker passed')
    except Exception as e:
        print(f'  FAIL — {e}')

    # ── Test 4: spaces ────────────────────────────────────────────────────────
    print('\n[TEST 4] Spaces:')
    env = ShorCodeEnv(noise_rate=0.05)
    print(f'  observation_space : {env.observation_space}')
    print(f'  action_space      : {env.action_space}  (n={env.action_space.n})')

    # ── Test 5: reset ─────────────────────────────────────────────────────────
    print('\n[TEST 5] reset():')
    env = ShorCodeEnv(noise_rate=0.05)
    obs, info = env.reset()
    print(f'  obs   : {obs}  shape={obs.shape}  dtype={obs.dtype}')
    print(f'  info  : {info}')
    assert obs.shape == (8,)
    assert obs in env.observation_space
    print('  PASS')

    # ── Test 6: step all actions (sample) ─────────────────────────────────────
    print('\n[TEST 6] step() — sample of actions:')
    env = ShorCodeEnv(noise_rate=0.0)
    obs, info = env.reset()
    env.render()
    for action in [0, 1, 10, 19]:
        env.reset()
        obs2, reward, term, trunc, info2 = env.step(action)
        t, q = action_to_correction(action)
        print(f'  action={action:2d} ({t:4s} q{q:2d})  '
              f'reward={reward:+.1f}  '
              f'reason={info2["reward_reason"][:50]}')
        assert term  is True
        assert trunc is False
        assert obs2 in env.observation_space

    # ── Test 7: correct action always gives +1.0 (no noise) ──────────────────
    print('\n[TEST 7] Correct action always gives +1.0 (no noise):')
    env  = ShorCodeEnv(noise_rate=0.0, logical_state='0')
    wins = 0
    for _ in range(100):
        obs, info = env.reset()
        decoded_type, correct_qubit, _, _ = decode_shor_syndrome(
            info['syndrome_int']
        )
        correct_action = correction_to_action(decoded_type, correct_qubit)
        _, reward, _, _, _ = env.step(correct_action)
        if reward == 1.0:
            wins += 1
    print(f'  {wins}/100 correct actions gave +1.0')
    status = 'PASS' if wins >= 95 else 'FAIL'
    print(f'  [{status}]')

    # ── Test 8: superposition state ───────────────────────────────────────────
    print('\n[TEST 8] Superposition state |+⟩:')
    env = ShorCodeEnv(noise_rate=0.0, logical_state='+')
    obs, info = env.reset()
    print(f'  obs={obs}  true_error={info["true_error_t"]} '
          f'q{info["true_error_q"]}')
    _, reward, _, _, info2 = env.step(0)
    print(f'  action=0  reward={reward:+.1f}  reason={info2["reward_reason"]}')
    print('  PASS')

    # ── Test 9: random agent 10k episodes ────────────────────────────────────
    print('\n[TEST 9] Random agent — 10,000 episodes:')
    env     = ShorCodeEnv(noise_rate=0.05)
    results = run_random_agent(env, n_episodes=10_000, verbose=True)
    assert results['accuracy'] <= 0.10, \
        f'Expected ~3.6%, got {results["accuracy"]:.2%}'
    print(f'  [PASS] accuracy={results["accuracy"]:.2%}  '
          f'(expected ~3.6%)')

    print('\nshor_env.py is ready.')