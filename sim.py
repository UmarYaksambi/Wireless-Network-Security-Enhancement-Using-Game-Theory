import streamlit as st
import random
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Defender vs Attacker Strategy Game", layout="wide")

# --- SETUP ---
st.title("🛡️ Defender vs 💣 Attacker: Strategic Game Simulation")

# Placeholder areas
status_placeholder = st.empty()
log_placeholder = st.empty()

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

# --- INITIAL STATES ---
log = []
attacker_score = 0
defender_score = 0
current_defense = None
previous_attack_outcomes = {atk: [] for atk in attack_strategies}


def choose_defense():
    """Choose defense based on max hit_rate and min resource."""
    return sorted(defense_strategies, key=lambda x: (-x['hit_rate'], x['resource']))[0]


def choose_attack():
    """Choose attack based on least success in past (try something new)."""
    attack_scores = {
        atk: sum(previous_attack_outcomes[atk]) / len(previous_attack_outcomes[atk]) if previous_attack_outcomes[atk] else 0
        for atk in attack_strategies
    }
    return min(attack_scores, key=attack_scores.get)


# --- SIMULATION LOOP ---
for step in range(simulation_steps):
    logs_this_step = [f"🔁 Step {step + 1}"]

    # Defender updates strategy periodically
    if step % strategy_change_interval == 0 or current_defense is None:
        current_defense = choose_defense()
        logs_this_step.append(f"🛡️ Defender chose strategy: {current_defense['name']} (Hit Rate: {current_defense['hit_rate']}, Resource: {current_defense['resource']})")

    # Attacker chooses blindly
    attacker_choice = choose_attack()
    logs_this_step.append(f"💣 Attacker tried: {attacker_choice}")

    # Evaluate outcome (random chance based on defender's hit rate)
    success_chance = random.random()
    if success_chance > current_defense["hit_rate"]:
        logs_this_step.append("🔥 Attack SUCCESSFUL!")
        attacker_score += 1
        previous_attack_outcomes[attacker_choice].append(1)
    else:
        logs_this_step.append("🛡️ Attack BLOCKED.")
        defender_score += 1
        previous_attack_outcomes[attacker_choice].append(0)

    # Update logs
    log.extend(logs_this_step)
    log_placeholder.code("\n".join(log[-10:]))

    # Show current status
    with status_placeholder.container():
        st.markdown(f"### 📊 Scores")
        col1, col2 = st.columns(2)
        col1.metric("Defender Score", defender_score)
        col2.metric("Attacker Score", attacker_score)

        st.markdown("### ⚔️ Current Strategies")
        st.markdown(f"**Defender Strategy**: {current_defense['name']} (Hit Rate: {current_defense['hit_rate']}, Resource: {current_defense['resource']})")
        st.markdown(f"**Attacker Chose**: {attacker_choice}")

    time.sleep(1)
