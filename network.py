import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import time

st.set_page_config(page_title="IoT Jamming Simulation", layout="wide")

# --- SETUP ---
NUM_NODES = 6
frequencies = [1, 2, 3]
G = nx.erdos_renyi_graph(NUM_NODES, 0.5, seed=42)

for node in G.nodes:
    G.nodes[node]['frequency'] = random.choice(frequencies)
    G.nodes[node]['status'] = 'idle'  # idle / transmitting / jammed

# Layout placeholders
graph_placeholder = st.empty()
log_placeholder = st.empty()

# Sidebar: Dynamic jammed frequency selector
jammed_frequency = st.sidebar.selectbox("Select jammed frequency", frequencies)
st.sidebar.markdown(f"### 🚨 Attacker is jamming **frequency {jammed_frequency}**")

# --- MAIN SIMULATION LOOP ---
log = []
run_time = 40  # Simulate 40 steps

for t in range(run_time):
    logs_this_tick = []

    # Each node tries to transmit
    for node in G.nodes:
        freq = G.nodes[node]['frequency']

        if freq == jammed_frequency:
            G.nodes[node]['status'] = 'jammed'
            logs_this_tick.append(f"[{t}] ❌ Node {node} BLOCKED (Freq {freq})")
        else:
            G.nodes[node]['status'] = 'transmitting'
            logs_this_tick.append(f"[{t}] ✅ Node {node} sent data (Freq {freq})")

    # Visualize the graph
    color_map = []
    for node in G.nodes:
        status = G.nodes[node]['status']
        if status == 'transmitting':
            color_map.append('green')
        elif status == 'jammed':
            color_map.append('red')
        else:
            color_map.append('gray')

    fig, ax = plt.subplots()
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=color_map, with_labels=False, ax=ax)
    # nx.draw_networkx_labels(G, pos, labels={n: f"{n}\nF{G.nodes[n]['frequency']}" for n in G.nodes}, ax=ax)p
    # This label is messing up node naming

    graph_placeholder.pyplot(fig)
    plt.close(fig)  # ✅ Prevent memory overload

    # Display logs
    log.extend(logs_this_tick)
    log_placeholder.code("\n".join(log[-10:]))  # Last 10 logs

    time.sleep(1)  # Real-time pause