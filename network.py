import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time

st.set_page_config(page_title="Realistic Wireless Jamming & Defense: Game Theoretic Models", layout="wide")

# --- SETUP ---
st.title("🛰️ Realistic Wireless Jamming & Defense: Game Theoretic Models")
NUM_NODES = 6
frequencies = [1, 2, 3]
G = nx.erdos_renyi_graph(NUM_NODES, 0.5, seed=42)
for node in G.nodes:
    G.nodes[node]['frequency'] = random.choice(frequencies)
    G.nodes[node]['status'] = 'idle'

# --- STRATEGY SPACES ---
attacker_strategies = ["broadband", "sweep", "reactive"]
defender_strategies = ["hop", "detect_and_switch", "stay"]

# --- STREAMLIT CONTROLS ---
mode = st.sidebar.radio("Choose Game Model", ["Bayesian Game", "Repeated Game"])
st.sidebar.markdown("---")
run_sim = st.sidebar.button("Start Simulation")
st.sidebar.markdown("Use 'Hop' or 'Detect & Switch' to dodge jamming.")

# --- SIMULATION SETTINGS ---
attack_cost = {"broadband": 3, "sweep": 2, "reactive": 1.5}
defense_cost = {"hop": 1, "detect_and_switch": 0.5, "stay": 0}

attacker_belief = {d: 1/len(defender_strategies) for d in defender_strategies}
defender_belief = {a: 1/len(attacker_strategies) for a in attacker_strategies}

# --- SCORE STATE ---
if "a_score" not in st.session_state:
    st.session_state.a_score = 0
if "d_score" not in st.session_state:
    st.session_state.d_score = 0

log_placeholder = st.empty()
graph_placeholder = st.empty()
status_placeholder = st.empty()

# --- UTILITY FUNCTION ---
def payoff(attacker, defender, success):
    if attacker not in attack_cost:
        attack_cost[attacker] = 2  # default
    if defender not in defense_cost:
        defense_cost[defender] = 1  # default
    ap = 5 if success else 0
    dp = 5 if not success else 0
    return ap - attack_cost[attacker], dp - defense_cost[defender]

# --- SIMULATION LOOP ---
if run_sim:
    logs = []
    for step in range(30):
        logs.append(f"\nStep {step+1}")

        # --- STRATEGY SELECTION ---
        if mode == "Bayesian Game":
            atk = max(attacker_belief, key=attacker_belief.get)
            dfd = max(defender_belief, key=defender_belief.get)
        else:
            atk = random.choice(attacker_strategies)
            dfd = random.choice(defender_strategies)

        logs.append(f"🛑 Attacker strategy: {atk}")
        logs.append(f"🛡️ Defender strategy: {dfd}")

        # --- ATTACK DECISION ---
        if atk == "broadband":
            jammed_freqs = frequencies
        elif atk == "sweep":
            jammed_freqs = [random.choice(frequencies)]
        elif atk == "reactive":
            active = {G.nodes[n]['frequency'] for n in G.nodes}
            jammed_freqs = [random.choice(list(active))] if active else []
        else:
            jammed_freqs = []

        # --- DEFENSE ACTION ---
        for node in G.nodes:
            if dfd == "hop":
                G.nodes[node]['frequency'] = random.choice([f for f in frequencies if f != G.nodes[node]['frequency']])
            elif dfd == "detect_and_switch" and G.nodes[node]['frequency'] in jammed_freqs:
                G.nodes[node]['frequency'] = random.choice([f for f in frequencies if f != G.nodes[node]['frequency']])

        # --- OUTCOME EVALUATION ---
        success = 0
        for node in G.nodes:
            f = G.nodes[node]['frequency']
            if f in jammed_freqs:
                G.nodes[node]['status'] = 'jammed'
                logs.append(f"❌ Node {node} jammed on Freq {f}")
            else:
                G.nodes[node]['status'] = 'ok'
                logs.append(f"✅ Node {node} transmitted on Freq {f}")
                success += 1

        # --- SCORE ---
        a_reward, d_reward = payoff(atk, dfd, success <= 2)
        st.session_state.a_score += a_reward
        st.session_state.d_score += d_reward

        # --- UI ---
        with status_placeholder.container():
            st.markdown("### 📊 Scores")
            col1, col2 = st.columns(2)
            col1.metric("Defender Score", round(st.session_state.d_score, 2))
            col2.metric("Attacker Score", round(st.session_state.a_score, 2))

        fig, ax = plt.subplots()
        pos = nx.spring_layout(G, seed=42)
        color_map = ['red' if G.nodes[n]['status'] == 'jammed' else 'green' for n in G.nodes]
        nx.draw(G, pos, node_color=color_map, with_labels=True, ax=ax)
        labels = {n: f"F{G.nodes[n]['frequency']}" for n in G.nodes}
        graph_placeholder.pyplot(fig)
        plt.close(fig)

        log_placeholder.code("\n".join(logs[-15:]))
        time.sleep(1)