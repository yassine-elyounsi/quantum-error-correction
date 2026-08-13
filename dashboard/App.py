"""
QuantumGuard — Live d=3 Surface Code Decoder Dashboard
=======================================================
Inject depolarizing errors, watch the stabilizers fire, watch the trained DDQN
agent decode step by step (with its Q-values), and compare against MWPM.

Run from your project root so `src/` and `reinforcement/` are importable:
    streamlit run dashboard/app.py
"""

import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.Demo_env import DemoSurfaceCodeEnv, MWPMDecoder, action_label
from dashboard.Surface_render import draw_lattice, draw_qvalues

st.set_page_config(page_title="QuantumGuard · d=3 Decoder", layout="wide")

st.markdown("""
<style>
.stButton>button {border-radius:10px; font-weight:600; padding:0.5rem 0.9rem;}
button[kind="primary"], button[data-testid="baseButton-primary"]{
    background:linear-gradient(90deg,#7c3aed,#ec4899); border:0; color:#fff;}
button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover{filter:brightness(1.06);}
div[data-testid="stMetricValue"]{font-size:1.35rem;}
.movelog{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
         padding:10px 12px;max-height:230px;overflow-y:auto;font-family:ui-monospace,monospace;font-size:12.5px;}
.movelog .row{padding:2px 0;color:#334155;}
.movelog .x{color:#2563eb;font-weight:700;}
.movelog .z{color:#d97706;font-weight:700;}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ── agent loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained agent…")
def load_agent(project_root, checkpoint, device):
    root = os.path.abspath(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from reinforcement.surface.ddqn_agent import DDQNAgent
    agent = DDQNAgent(in_channels=6, grid_size=7, n_actions=19,
                      buffer_capacity=1000, device=device)
    ckpt = checkpoint if os.path.isabs(checkpoint) else os.path.join(root, checkpoint)
    agent.load(ckpt)
    agent.online_net.eval()
    return agent


def greedy_q(agent, obs):
    q = np.asarray(agent.get_q_values(obs), dtype=float)
    return q, int(np.argmax(q))


# ── session helpers ───────────────────────────────────────────────────────────
def get_env():
    if "env" not in st.session_state:
        st.session_state.env = DemoSurfaceCodeEnv(distance=3, max_steps=50)
        st.session_state.mwpm = MWPMDecoder(st.session_state.env)
    return st.session_state.env


def make_frame(env, q, action, info):
    return dict(render=env.get_render_state(),
                q=(None if q is None else list(map(float, q))),
                action=action, info=dict(info))


def start_episode_from_env(env, info):
    weight = info["syndrome_weight"]
    if weight == 0:                                   # trivial or undetectable logical
        logical = env.is_logical_error()
        info = {**info, "corrected": (not logical), "logical_error": logical}
    st.session_state.frames = [make_frame(env, None, None, info)]
    st.session_state.idx = 0
    st.session_state.done = (weight == 0)
    mwpm = st.session_state.get("mwpm")
    if mwpm and mwpm.ok:
        st.session_state.mwpm_result = mwpm.decode(env._x_error.copy(), env._z_error.copy())
    else:
        st.session_state.mwpm_result = None


def inject_random(env, p, seed):
    if seed is not None:
        obs, info = env.reset(p=p, seed=int(seed))
    else:                                             # resample until something fires
        for _ in range(80):
            obs, info = env.reset(p=p)
            if info["syndrome_weight"] > 0:
                break
    start_episode_from_env(env, info)


def inject_manual(env, xe, ze):
    obs, info = env.reset(error=(xe, ze))
    start_episode_from_env(env, info)


def agent_step(env, agent):
    if st.session_state.done:
        return
    q, a = greedy_q(agent, env._get_obs())
    _, _, term, trunc, info = env.step(a)
    st.session_state.frames.append(make_frame(env, q, a, info))
    st.session_state.idx = len(st.session_state.frames) - 1
    st.session_state.done = term or trunc


def auto_solve(env, agent):
    guard = 0
    while not st.session_state.done and guard < env.max_steps + 2:
        agent_step(env, agent); guard += 1


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Setup")
project_root = st.sidebar.text_input("Project root", value=".")
checkpoint   = st.sidebar.text_input("Checkpoint", value="checkpoints_stim_final/d3_best.pt")
device       = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)

agent, agent_err = None, None
try:
    agent = load_agent(project_root, checkpoint, device)
except Exception as e:
    agent_err = e

env = get_env()

st.sidebar.markdown("---")
st.sidebar.subheader("Noise model")
st.sidebar.caption("Perfect encoding · perfect measurement · depolarizing on data qubits only")
mode = st.sidebar.radio("Error injection", ["Random depolarizing", "Manual (place errors)"])
show_mwpm = st.sidebar.checkbox("Compare with MWPM", value=True)

if mode == "Random depolarizing":
    p = st.sidebar.slider("Depolarizing probability p", 0.01, 0.30, 0.10, 0.01)
    use_seed = st.sidebar.checkbox("Fixed seed", value=False)
    seed = st.sidebar.number_input("Seed", 0, 10_000, 0, 1) if use_seed else None
    if st.sidebar.button("🎲 Inject new error", use_container_width=True, type="primary"):
        inject_random(env, p, seed); st.rerun()
else:
    st.sidebar.caption("Pick a Pauli for each of the 9 data qubits")
    cols = st.sidebar.columns(3)
    sel = [cols[i % 3].selectbox(f"q{i}", ["I", "X", "Y", "Z"], key=f"man_{i}") for i in range(9)]
    if st.sidebar.button("💉 Inject manual error", use_container_width=True, type="primary"):
        xe = np.array([1 if s in ("X", "Y") else 0 for s in sel], np.int8)
        ze = np.array([1 if s in ("Z", "Y") else 0 for s in sel], np.int8)
        inject_manual(env, xe, ze); st.rerun()

if "frames" not in st.session_state:
    inject_random(env, 0.10, None)


# ── header / status ───────────────────────────────────────────────────────────
top = st.columns([3, 1])
with top[0]:
    st.title("QuantumGuard — d = 3 Surface Code Decoder")
with top[1]:
    if agent is not None:
        st.markdown("<div style='text-align:right;margin-top:18px'>"
                    "<span class='tag' style='background:#dcfce7;color:#166534'>● agent loaded</span> "
                    f"<span class='tag' style='background:#e0e7ff;color:#3730a3'>{device}</span></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:right;margin-top:18px'>"
                    "<span class='tag' style='background:#fee2e2;color:#991b1b'>● agent not loaded</span></div>",
                    unsafe_allow_html=True)

if agent_err is not None:
    st.error(
        f"Could not load the agent from `{checkpoint}` under `{project_root}`.\n\n"
        f"`{type(agent_err).__name__}: {agent_err}`\n\n"
        "Run `streamlit run dashboard/app.py` from your project root (the folder that "
        "contains `reinforcement/` and `src/`), or set **Project root** accordingly. "
        "The lattice and MWPM still work below; agent steps are disabled."
    )


# ── controls ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    if st.button("▶ Agent · 1 move", use_container_width=True,
                 disabled=(agent is None or st.session_state.done)):
        agent_step(env, agent); st.rerun()
with c2:
    if st.button("⏭ Auto-solve", use_container_width=True, type="primary",
                 disabled=(agent is None or st.session_state.done)):
        auto_solve(env, agent); st.rerun()
with c3:
    if st.button("↺ Replay error", use_container_width=True):
        inject_manual(env, env._x_error.copy(), env._z_error.copy()); st.rerun()
with c4:
    n = len(st.session_state.frames)
    if n > 1:
        st.session_state.idx = st.slider("Replay step", 0, n - 1,
                                         min(st.session_state.idx, n - 1))
    else:
        st.caption("Inject an error, then press **Auto-solve** to watch the agent decode.")

frame = st.session_state.frames[st.session_state.idx]
rs = frame["render"]
info = frame["info"]
weight = rs["weight"]
is_last = st.session_state.idx == len(st.session_state.frames) - 1


# ── main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    fig = draw_lattice(rs)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    if is_last and st.session_state.done:
        if info.get("corrected"):
            st.success("✅ **Corrected** — syndrome cleared, logical qubit preserved.")
        elif info.get("logical_error"):
            st.error("❌ **Logical error** — the logical qubit flipped.")
        else:
            st.warning("⚠️ **Did not converge** within the step budget.")
    else:
        st.info(f"Syndrome weight **{weight}** · step **{info['step']}** — "
                f"{'press Auto-solve' if info['step']==0 else 'decoding…'}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Syndrome weight", weight)
    m2.metric("Step", info["step"])
    m3.metric("Action", action_label(frame["action"]))

    if frame["q"] is not None:
        st.markdown("**Agent Q-values** · pink = chosen")
        qfig = draw_qvalues(np.array(frame["q"]), chosen=frame["action"])
        st.pyplot(qfig, use_container_width=True)
        plt.close(qfig)
    else:
        st.caption("Q-values appear once the agent takes its first step.")

    nq = rs["n_qubits"]
    ex = [i for i in range(nq) if rs["x_error"][i]]
    ez = [i for i in range(nq) if rs["z_error"][i]]
    st.markdown(f"**Injected error** — X on {ex or '—'}, Z on {ez or '—'}")


# ── move log + MWPM ───────────────────────────────────────────────────────────
log_col, cmp_col = st.columns([3, 2])

with log_col:
    st.markdown("**Move log**")
    rows = ""
    for t, fr in enumerate(st.session_state.frames):
        if fr["action"] is None:
            rows += f"<div class='row'>{t:>2}. inject · syndrome weight {fr['render']['weight']}</div>"
        else:
            a = fr["action"]; lab = action_label(a)
            conf = fr["q"][a] if fr["q"] is not None else 0.0
            cls = "x" if a < nq else ("z" if a < 2 * nq else "")
            rows += (f"<div class='row'>{t:>2}. agent · <span class='{cls}'>{lab}</span> "
                     f"· Q={conf:+.2f} · weight {fr['render']['weight']}</div>")
    st.markdown(f"<div class='movelog'>{rows}</div>", unsafe_allow_html=True)

with cmp_col:
    if show_mwpm:
        st.markdown("**Agent vs MWPM — same error**")
        a, b = st.columns(2)
        with a:
            st.caption("RL agent")
            if st.session_state.done and is_last:
                if info.get("corrected"):
                    st.success("Corrected ✅")
                elif info.get("logical_error"):
                    st.error("Logical ❌")
                else:
                    st.warning("No converge ⚠️")
            else:
                st.caption("run Auto-solve")
        with b:
            st.caption("MWPM")
            res = st.session_state.get("mwpm_result")
            if not st.session_state.mwpm.ok:
                st.caption("PyMatching n/a")
            elif res is None:
                st.caption("—")
            else:
                ok, xc, zc = res
                if ok:
                    st.success("Corrected ✅")
                else:
                    st.error("Logical ❌")

st.caption(
    "Code-capacity model: perfect encoding and measurement, depolarizing noise on data "
    "qubits. Observations use the exact training layout (in-distribution for the network); "
    "logical failure is checked against the true weight-3 logical operators and agrees "
    "with MWPM's verdict on thousands of errors."
)