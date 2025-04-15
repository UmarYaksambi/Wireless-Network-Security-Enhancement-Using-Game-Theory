import streamlit as st
import random
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Defender vs Attacker Strategy Game", layout="wide")

# --- SETUP ---
st.title("🛡️ Defender vs 💣 Attacker: Strategic Game Simulation")

# --- SESSION STATE INIT ---
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.attacker_score = 0
    st.session_state.defender_score = 0
    st.session_state.log = []
    st.session_state.current_defense = None
    st.session_state.previous_attack_outcomes = {atk: [] for atk in ["Brute Force", "Phishing", "MITM", "DoS"]}
    st.session_state.defense_history = []
    st.session_state.attack_history = []

# --- RESET BUTTON ---
if st.sidebar.button("🔄 Restart Simulation"):
    st.session_state.step = 0
    st.session_state.attacker_score = 0
    st.session_state.defender_score = 0
    st.session_state.log = []
    st.session_state.current_defense = None
    st.session_state.previous_attack_outcomes = {atk: [] for atk in ["Brute Force", "Phishing", "MITM", "DoS"]}
    st.session_state.defense_history = []
    st.session_state.attack_history = []
    st.rerun()

# Placeholder areas
status_placeholder = st.container()
log_placeholder = st.container()
chart_placeholder = st.container()

# --- PARAMETERS ---
simulation_steps = 40
strategy_change_interval = 5  # Defender changes strategy every N steps

# --- DEFENSE STRATEGIES ---
defense_strategies = [
    {"name": "Firewall", "resource": 3, "hit_rate": 0.8},
    {"name": "Encryption", "resource": 2, "hit_rate": 0.6},
    {"name": "Traffic Analysis", "resource": 1, "hit_rate": 0.4},
    {"name": "Multi-factor", "resource": 4, "hit_rate": 0.9},
]

# --- ATTACK STRATEGIES ---
attack_strategies = ["Brute Force", "Phishing", "MITM", "DoS"]

# --- STACKELBERG DEFENSE STRATEGY ---
def choose_stackelberg_defense():
    expected_success = []
    for defense in defense_strategies:
        blocked = 0
        for atk in attack_strategies:
            hit_rate = defense['hit_rate']
            blocked += hit_rate * (1 - sum(st.session_state.previous_attack_outcomes[atk]) / len(st.session_state.previous_attack_outcomes[atk]) if st.session_state.previous_attack_outcomes[atk] else 1)
        expected_success.append((blocked, defense))
    return sorted(expected_success, key=lambda x: (-x[0], x[1]['resource']))[0][1]

# --- ATTACK STRATEGY ---
def choose_attack():
    attack_scores = {
        atk: sum(st.session_state.previous_attack_outcomes[atk]) / len(st.session_state.previous_attack_outcomes[atk]) if st.session_state.previous_attack_outcomes[atk] else 0
        for atk in attack_strategies
    }
    return min(attack_scores, key=attack_scores.get)

# --- SIMULATION LOOP (1 step per rerun) ---
if st.session_state.step < simulation_steps:
    logs_this_step = [f"🔁 Step {st.session_state.step + 1}"]

    if st.session_state.step % strategy_change_interval == 0 or st.session_state.current_defense is None:
        st.session_state.current_defense = choose_stackelberg_defense()
        logs_this_step.append(f"🛡️ Defender chose strategy: {st.session_state.current_defense['name']} (Hit Rate: {st.session_state.current_defense['hit_rate']}, Resource: {st.session_state.current_defense['resource']})")

    attacker_choice = choose_attack()
    logs_this_step.append(f"💣 Attacker tried: {attacker_choice}")

    success_chance = random.random()
    if success_chance > st.session_state.current_defense["hit_rate"]:
        logs_this_step.append("🔥 Attack SUCCESSFUL!")
        st.session_state.attacker_score += 1
        st.session_state.previous_attack_outcomes[attacker_choice].append(1)
    else:
        logs_this_step.append("🛡️ Attack BLOCKED.")
        st.session_state.defender_score += 1
        st.session_state.previous_attack_outcomes[attacker_choice].append(0)

    st.session_state.defense_history.append(st.session_state.current_defense['name'])
    st.session_state.attack_history.append(attacker_choice)

    st.session_state.log.extend(logs_this_step)
    log_placeholder.code("\n".join(st.session_state.log[-10:]))

    # --- STATUS ---
    with status_placeholder:
        st.markdown(f"### 📊 Scores")
        col1, col2 = st.columns(2)
        col1.metric("Defender Score", st.session_state.defender_score)
        col2.metric("Attacker Score", st.session_state.attacker_score)

        st.markdown("### ⚔️ Current Strategies")
        st.markdown(f"**Defender Strategy**: {st.session_state.current_defense['name']} (Hit Rate: {st.session_state.current_defense['hit_rate']}, Resource: {st.session_state.current_defense['resource']})")
        st.markdown(f"**Attacker Chose**: {attacker_choice}")

        if st.session_state.attacker_score > st.session_state.defender_score + 5:
            st.error("⚠️ Critical: Attacker is leading significantly!")
        elif st.session_state.defender_score > st.session_state.attacker_score + 5:
            st.success("✅ Defender has strong control.")

    st.session_state.step += 1
    time.sleep(1)
    st.rerun()
else:
    with status_placeholder:
        st.markdown(f"### 📊 Final Scores")
        col1, col2 = st.columns(2)
        col1.metric("Defender Score", st.session_state.defender_score)
        col2.metric("Attacker Score", st.session_state.attacker_score)

    with log_placeholder:
        st.markdown("### 🧾 Final Log")
        st.code("\n".join(st.session_state.log))

    with chart_placeholder:
        if len(st.session_state.defense_history) > 1:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(st.session_state.defense_history, label="Defense", marker='o')
            ax.plot(st.session_state.attack_history, label="Attack", marker='x')
            ax.set_title("Strategy Choices Over Time")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Strategy")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
