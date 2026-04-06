

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.circuits.repetition_code import (
    run_syndrome_extraction,
    decode_repetition_syndrome,
    REPETITION_SYNDROME_TABLE,
)


# =============================================================================
# ACTION MAP
# =============================================================================
# Maps action integer → correction qubit (-1 = no correction)
#
#   Action 0 → no correction      (correct when syndrome = 00)
#   Action 1 → X on qubit 0       (correct when syndrome = 01)
#   Action 2 → X on qubit 1       (correct when syndrome = 11)
#   Action 3 → X on qubit 2       (correct when syndrome = 10)

ACTION_TO_QUBIT = {
    0: -1,   # no correction
    1:  0,   # correct qubit 0
    2:  1,   # correct qubit 1
    3:  2,   # correct qubit 2
}

# Reverse map — qubit → action (used to find the correct action)
QUBIT_TO_ACTION = {v: k for k, v in ACTION_TO_QUBIT.items()}

N_ACTIONS = 4


# =============================================================================
# REWARD TABLE
# =============================================================================
# Built directly from REPETITION_SYNDROME_TABLE.
# Maps (syndrome_int, action) → reward
#
# Logic:
#   syndrome = 00 (no error):
#       action = 0 (no correction)  → +1.0   correct
#       action = 1,2,3              → -1.0   unnecessary correction
#
#   syndrome = 01 (error on q0):
#       action = 1 (correct q0)     → +1.0   correct
#       action = 0 (no correction)  → -0.5   missed error
#       action = 2,3                → -0.5   wrong qubit
#
#   syndrome = 11 (error on q1):
#       action = 2 (correct q1)     → +1.0   correct
#       action = 0                  → -0.5
#       action = 1,3                → -0.5
#
#   syndrome = 10 (error on q2):
#       action = 3 (correct q2)     → +1.0   correct
#       action = 0                  → -0.5
#       action = 1,2                → -0.5

def _build_reward_table():
    """
    Build the complete reward table from the syndrome table.

    Returns
    -------
    reward_table : dict   (syndrome_int, action) -> (reward, reason)
    """
    table = {}

    for syndrome_int, (error_type, correct_qubit, description) in \
            REPETITION_SYNDROME_TABLE.items():

        for action in range(N_ACTIONS):
            action_qubit = ACTION_TO_QUBIT[action]

            if error_type == 'None':
                # No error detected
                if action == 0:
                    table[(syndrome_int, action)] = (
                        +1.0, 'correct: no error detected, no correction applied'
                    )
                else:
                    table[(syndrome_int, action)] = (
                        -1.0, f'wrong: unnecessary correction on q{action_qubit}'
                    )

            else:
                # Error detected — correct_qubit is the right answer
                if action_qubit == correct_qubit:
                    table[(syndrome_int, action)] = (
                        +1.0, f'correct: X on q{correct_qubit} matches syndrome'
                    )
                elif action == 0:
                    table[(syndrome_int, action)] = (
                        -0.5, f'wrong: error on q{correct_qubit} missed, no correction'
                    )
                else:
                    table[(syndrome_int, action)] = (
                        -0.5, f'wrong: corrected q{action_qubit}, error was on q{correct_qubit}'
                    )

    return table


REWARD_TABLE = _build_reward_table()


# =============================================================================
# ENVIRONMENT
# =============================================================================

class RepetitionCodeEnv(gym.Env):
    """
    Gymnasium environment for the 3-qubit bit-flip repetition code.

    The agent observes a 2-bit syndrome vector and must select
    which qubit to correct (or do nothing).
    Reward is determined entirely by the syndrome table — no circuit
    is built in step() and no fidelity is computed.

    Parameters
    ----------
    noise_rate    : float
        Depolarising noise per gate [0.0, 1.0].
    noise_type    : str
        'depolarising' or 'x_only'.
    logical_state : str or tuple
        Logical state to encode each episode.
        '0', '1', '+', '-', (theta, phi), or 'random'.
        'random' picks uniformly from ['0', '1'] each episode.

    Observation Space
    -----------------
    MultiBinary(2) — syndrome vector [s0, s1]

    Action Space
    ------------
    Discrete(4)
        0 → no correction
        1 → X on qubit 0
        2 → X on qubit 1
        3 → X on qubit 2

    Reward
    ------
    +1.0   action matches what syndrome table prescribes
    -0.5   error detected but wrong qubit corrected, or missed entirely
    -1.0   no error detected but correction applied anyway
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
        self.observation_space = spaces.MultiBinary(2)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        # Episode state — set by reset()
        self._syndrome_int         = None
        self._syndrome_str         = None
        self._syndrome_vec         = None
        self._true_error_qubit     = None
        self._true_error_type      = None
        self._current_logical_state = None

        # Cumulative statistics
        self._episode_count  = 0
        self._total_reward   = 0.0
        self._correct_count  = 0

    # ── RESET ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Start a new episode.

        Randomly injects an error (or none), runs noisy syndrome
        extraction, and returns the syndrome vector as observation.

        Returns
        -------
        observation : np.ndarray   shape (2,)  dtype uint8
        info        : dict
        """
        super().reset(seed=seed)

        # Logical state for this episode
        if self.logical_state == 'random':
            self._current_logical_state = self.np_random.choice(['0', '1'])
        else:
            self._current_logical_state = self.logical_state

        # Randomly inject an error on one qubit or nothing
        # Balanced injection ensures agent sees all syndromes equally
        choice = self.np_random.integers(0, 4)   # 0,1,2,3
        self._true_error_qubit = None if choice == 0 else int(choice - 1)
        self._true_error_type  = (
            'X' if self._true_error_qubit is not None else 'None'
        )

        # Run syndrome extraction (1 shot)
        syndrome_int, syndrome_str, _ = run_syndrome_extraction(
            logical_state=self._current_logical_state,
            error_qubit=self._true_error_qubit,
            p=self.noise_rate,
            shots=1,
            noise_type=self.noise_type
        )

        self._syndrome_int = syndrome_int
        self._syndrome_str = syndrome_str
        self._syndrome_vec = np.array(
            [int(b) for b in syndrome_str], dtype=np.float32
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
        Reward is looked up directly from the syndrome reward table.
        No quantum circuit is executed here.

        Parameters
        ----------
        action : int   0=no correction, 1=X q0, 2=X q1, 3=X q2

        Returns
        -------
        observation  : np.ndarray   shape (2,)
        reward       : float
        terminated   : bool         always True (1 step per episode)
        truncated    : bool         always False
        info         : dict
        """
        if self._syndrome_vec is None:
            raise RuntimeError('Call reset() before step().')

        assert self.action_space.contains(action), \
            f'Invalid action {action}. Must be in [0, {N_ACTIONS - 1}]'

        # Look up reward from table
        if isinstance(action, np.ndarray):
           action = action.item()
        action = int(action)
        reward, reason = REWARD_TABLE[(self._syndrome_int, action)]

        # Decode syndrome to know what the correct action was
        decoded_type, correct_qubit, description = decode_repetition_syndrome(
            self._syndrome_int
        )
        correct_action = QUBIT_TO_ACTION.get(correct_qubit, 0)

        # Update statistics
        self._total_reward  += reward
        if reward > 0:
            self._correct_count += 1

        observation = self._syndrome_vec.copy()
        info = {
            'action'        : action,
            'action_qubit'  : ACTION_TO_QUBIT[action],
            'correct_action': correct_action,
            'correct_qubit' : correct_qubit,
            'decoded_type'  : decoded_type,
            'reward'        : reward,
            'reward_reason' : reason,
            'syndrome_int'  : self._syndrome_int,
            'syndrome_str'  : self._syndrome_str,
            'true_error_q'  : self._true_error_qubit,
            'true_error_t'  : self._true_error_type,
            'logical_state' : self._current_logical_state,
        }

        return observation, reward, True, False, info

    # ── RENDER ────────────────────────────────────────────────────────────────

    def render(self):
        """Print current episode state."""
        if self._syndrome_vec is None:
            print('Environment not reset yet. Call reset() first.')
            return

        decoded_type, correct_qubit, _ = decode_repetition_syndrome(
            self._syndrome_int
        )
        correct_action = QUBIT_TO_ACTION.get(correct_qubit, 0)

        print(f'\n  RepetitionCodeEnv')
        print(f'  {"─"*35}')
        print(f'  Logical state    : |{self._current_logical_state}⟩')
        print(f'  Noise rate       : {self.noise_rate*100:.1f}%')
        print(f'  Syndrome vector  : {self._syndrome_vec}  '
              f'(binary={self._syndrome_str}, int={self._syndrome_int})')
        print(f'  True error       : {self._true_error_type} '
              f'on q{self._true_error_qubit}')
        print(f'  Correct action   : {correct_action}  '
              f'(correct q{correct_qubit})')
        print(f'  {"─"*35}')

    # ── STATS ─────────────────────────────────────────────────────────────────

    def get_stats(self):
        """Return cumulative statistics across all episodes."""
        ep = max(self._episode_count, 1)
        return {
            'episodes'    : self._episode_count,
            'total_reward': round(self._total_reward, 2),
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

    A random agent picks uniformly from [0,1,2,3].
    Expected accuracy is ~25% by chance.
    Any trained RL agent must significantly beat this.

    Parameters
    ----------
    env        : RepetitionCodeEnv
    n_episodes : int
    verbose    : bool

    Returns
    -------
    results : dict
    """
    env.reset_stats()

    reward_history = []
    correct        = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        action    = env.action_space.sample()   # random action
        _, reward, _, _, info = env.step(action)

        reward_history.append(reward)
        if reward > 0:
            correct += 1

        if verbose and n_episodes >= 5 and (ep + 1) % (n_episodes // 5) == 0:
            window     = reward_history[-(n_episodes // 5):]
            mean_r     = np.mean(window)
            acc_so_far = correct / (ep + 1)
            print(f'  ep={ep+1:6d}  '
                  f'mean_reward={mean_r:+.3f}  '
                  f'accuracy={acc_so_far:.2%}')

    results = {
        'n_episodes'  : n_episodes,
        'mean_reward' : round(float(np.mean(reward_history)), 4),
        'std_reward'  : round(float(np.std(reward_history)),  4),
        'accuracy'    : round(correct / n_episodes,           4),
        'reward_dist' : {
            '+1.0': reward_history.count(1.0),
            '-0.5': reward_history.count(-0.5),
            '-1.0': reward_history.count(-1.0),
        }
    }

    if verbose:
        print(f'\n  {"─"*40}')
        print(f'  RANDOM AGENT — {n_episodes} episodes')
        print(f'  {"─"*40}')
        print(f'  Mean reward   : {results["mean_reward"]:+.4f}')
        print(f'  Std reward    : {results["std_reward"]:.4f}')
        print(f'  Accuracy      : {results["accuracy"]:.2%}')
        print(f'  Reward dist   : {results["reward_dist"]}')
        print(f'  {"─"*40}')
        print(f'  Expected ~25% accuracy (4 actions, 1 correct)')
        print(f'  Trained RL agent should reach >90%')

    return results


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':

    print('=' * 55)
    print('  repetition_env.py — self test')
    print('=' * 55)

    # ── Test 1: reward table ──────────────────────────────────────────────────
    print('\n[TEST 1] Reward Table:')
    for (syn, act), (rew, reason) in sorted(REWARD_TABLE.items()):
        print(f'  syndrome={syn:02b}({syn})  action={act}  '
              f'reward={rew:+.1f}  {reason}')

    # ── Test 2: env_checker ───────────────────────────────────────────────────
    print('\n[TEST 2] gymnasium.utils.env_checker:')
    from gymnasium.utils.env_checker import check_env
    env = RepetitionCodeEnv(noise_rate=0.05)
    try:
        check_env(env, warn=True)
        print('  PASS — env_checker passed')
    except Exception as e:
        print(f'  FAIL — {e}')

    # ── Test 3: spaces ────────────────────────────────────────────────────────
    print('\n[TEST 3] Spaces:')
    env = RepetitionCodeEnv(noise_rate=0.05)
    print(f'  observation_space : {env.observation_space}')
    print(f'  action_space      : {env.action_space}')

    # ── Test 4: reset ─────────────────────────────────────────────────────────
    print('\n[TEST 4] reset():')
    env = RepetitionCodeEnv(noise_rate=0.05)
    obs, info = env.reset()
    print(f'  obs   : {obs}  shape={obs.shape}  dtype={obs.dtype}')
    print(f'  info  : {info}')
    assert obs.shape == (2,)
    assert obs in env.observation_space
    print('  PASS')

    # ── Test 5: step all actions ──────────────────────────────────────────────
    print('\n[TEST 5] step() — all actions:')
    env = RepetitionCodeEnv(noise_rate=0.0)
    for action in range(4):
        obs, info = env.reset()
        obs2, reward, term, trunc, info2 = env.step(action)
        print(f'  action={action}  syndrome={obs}  '
              f'reward={reward:+.1f}  reason={info2["reward_reason"]}')
        assert term  is True
        assert trunc is False
        assert obs2 in env.observation_space
    print('  PASS')

    # ── Test 6: correct action always gets +1.0 ───────────────────────────────
    print('\n[TEST 6] Correct action always gives +1.0 (no noise):')
    env   = RepetitionCodeEnv(noise_rate=0.0, logical_state='0')
    wins  = 0
    for _ in range(100):
        obs, info = env.reset()
        env.render()
        correct_action = QUBIT_TO_ACTION.get(
            decode_repetition_syndrome(info['syndrome_int'])[1], 0
        )
        _, reward, _, _, _ = env.step(correct_action)
        if reward == 1.0:
            wins += 1
    print(f'  {wins}/100 correct actions gave +1.0')
    status = 'PASS' if wins == 100 else 'FAIL'
    print(f'  [{status}]')

    # ── Test 7: random agent 10k episodes ────────────────────────────────────
    print('\n[TEST 7] Random agent — 10,000 episodes:')
    env     = RepetitionCodeEnv(noise_rate=0.05)
    results = run_random_agent(env, n_episodes=10_000, verbose=True)
    assert 0.15 <= results['accuracy'] <= 0.40, \
        f'Expected ~25%, got {results["accuracy"]:.2%}'
    print(f'  [PASS] accuracy={results["accuracy"]:.2%}  '
          f'(expected ~25%)')

    print('\nrepetition_env.py is ready.')