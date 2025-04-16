import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import nashpy as nash
import numpy as np

st.set_page_config(page_title="IoT Jamming Simulation", layout="wide")

# --- SETUP ---
NUM_NODES = 6
frequencies = [1, 2, 3]
G = nx.erdos_renyi_graph(NUM_NODES, 0.5, seed=42)

for node in G.nodes:
    G.nodes[node]['frequency'] = random.choice(frequencies)
    G.nodes[node]['status'] = 'idle'  # idle / transmitting / compromised

# Layout placeholders
graph_placeholder = st.empty()
log_placeholder = st.empty()
score_placeholder = st.empty()

# --- SIDEBAR CONTROLS ---
attacker_strategies = ["constant", "random", "selective"]
defender_strategies = ["stay", "switch"]
strategy_mode = st.sidebar.selectbox("Strategy Mode", ["game-theory", "fixed", "mixed"])
fixed_attacker_strategy = st.sidebar.selectbox("Fixed Attacker Strategy", attacker_strategies)
auto_defend = st.sidebar.checkbox("Enable Auto Defender", value=True)

# Manual node frequency adjustment
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Change Node Frequency")
node_to_edit = st.sidebar.selectbox("Select Node", list(G.nodes))
new_freq = st.sidebar.selectbox("New Frequency", frequencies)
if st.sidebar.button("Update Frequency"):
    G.nodes[node_to_edit]['frequency'] = new_freq
    st.sidebar.success(f"Node {node_to_edit} set to frequency {new_freq}")

# --- GAME THEORY PAYOFF MATRIX ---
payoff_matrix_attacker = np.array([
    [2, -2],
    [0,  0],
    [3, -3]
])

payoff_matrix_defender = np.array([
    [-5, 5],
    [0,  0],
    [-7, 7]
])

payoffs = {
    ("constant", "stay"):     (2, -5),
    ("constant", "switch"):   (-2, 5),
    ("random", "stay"):       (0, 0),
    ("random", "switch"):     (0, 0),
    ("selective", "stay"):    (3, -7),
    ("selective", "switch"):  (-3, 7),
}

# Scores
if "attacker_score" not in st.session_state:
    st.session_state.attacker_score = 0
if "defender_score" not in st.session_state:
    st.session_state.defender_score = 0
if "current_defense" not in st.session_state:
    st.session_state.current_defense = {"name": "stay", "hit_rate": 0, "resource": 0}
if "attacker_choice" not in st.session_state:
    st.session_state.attacker_choice = "constant"

# --- GAME THEORY STRATEGY SELECTION ---
def get_nash_equilibrium():
    game = nash.Game(payoff_matrix_attacker, payoff_matrix_defender)
    equilibria = list(game.support_enumeration())
    if equilibria:
        atk_strategy, def_strategy = equilibria[0]  # Use first equilibrium
        atk_choice = attacker_strategies[np.argmax(atk_strategy)]
        def_choice = defender_strategies[np.argmax(def_strategy)]
        return atk_choice, def_choice
    return "random", "stay"

# --- MAIN SIMULATION LOOP ---
run_time = 40
log = []

for t in range(run_time):
    logs_this_tick = []

    # --- Attacker & Defender choose strategy ---
    if strategy_mode == "fixed":
        attacker_choice = fixed_attacker_strategy
        defender_choice = "switch" if auto_defend else "stay"
    elif strategy_mode == "mixed":
        attacker_choice = random.choices(attacker_strategies, weights=[0.3, 0.4, 0.3])[0]
        defender_choice = "switch" if auto_defend else "stay"
    else:
        attacker_choice, defender_choice = get_nash_equilibrium()

    st.session_state.attacker_choice = attacker_choice
    st.session_state.current_defense['name'] = defender_choice

    # Determine jammed frequency
    if attacker_choice == "constant":
        jammed_frequency = frequencies[0]
    elif attacker_choice == "random":
        jammed_frequency = random.choice(frequencies)
    elif attacker_choice == "selective":
        transmitting_freqs = {G.nodes[n]['frequency'] for n in G.nodes}
        jammed_frequency = random.choice(list(transmitting_freqs))

    logs_this_tick.append(f"[{t}] 🎯 Attacker uses {attacker_choice} jamming on frequency {jammed_frequency}")

    # --- Node transmission and payoff application ---
    for node in G.nodes:
        freq = G.nodes[node]['frequency']
        if freq == jammed_frequency:
            if defender_choice == "switch":
                new_freq = random.choice([f for f in frequencies if f != jammed_frequency])
                G.nodes[node]['frequency'] = new_freq
                logs_this_tick.append(f"[{t}] 🛡 Node {node} switched to Freq {new_freq}")
                freq = new_freq

        # Status Update
        if G.nodes[node]['frequency'] == jammed_frequency:
            G.nodes[node]['status'] = 'compromised'
            logs_this_tick.append(f"[{t}] ❌ Node {node} COMPROMISED on Freq {jammed_frequency}")
        else:
            G.nodes[node]['status'] = 'transmitting'
            logs_this_tick.append(f"[{t}] ✅ Node {node} sent data on Freq {G.nodes[node]['frequency']}")

        # Apply game theory payoff
        att_payoff, def_payoff = payoffs[(attacker_choice, defender_choice)]
        st.session_state.attacker_score += att_payoff
        st.session_state.defender_score += def_payoff

    # --- Scores and Strategies Display (Once) ---
    with score_placeholder.container():
        st.markdown(f"### 📊 Scores")
        col1, col2 = st.columns(2)
        col1.metric("Defender Score", st.session_state.defender_score)
        col2.metric("Attacker Score", st.session_state.attacker_score)

        st.markdown("### ⚔️ Current Strategies")
        st.markdown(f"**Defender Strategy**: {defender_choice}")
        st.markdown(f"**Attacker Chose**: {attacker_choice}")

        if st.session_state.attacker_score > st.session_state.defender_score + 5:
            st.error("⚠️ Critical: Attacker is leading significantly!")
        elif st.session_state.defender_score > st.session_state.attacker_score + 5:
            st.success("✅ Defender has strong control.")

    # --- Visualization ---
    color_map = ['red' if G.nodes[n]['status'] == 'compromised' else 'green' for n in G.nodes]
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=color_map, with_labels=True, ax=ax)
    labels = {n: f"      Freq {G.nodes[n]['frequency']}" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=10, verticalalignment='bottom')
    graph_placeholder.pyplot(fig)
    plt.close(fig)

    # Log Output
    log.extend(logs_this_tick)
    log_placeholder.code("\n".join(log[-10:]))

    time.sleep(1)
