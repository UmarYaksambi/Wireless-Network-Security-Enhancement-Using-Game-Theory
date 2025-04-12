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
st.sidebar.markdown(f"### 🚨 Attacker is jamming frequency {jammed_frequency}")

# Sidebar: Manual node frequency adjustment
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Change Node Frequency")
node_to_edit = st.sidebar.selectbox("Select Node", list(G.nodes))
new_freq = st.sidebar.selectbox("New Frequency", frequencies)
if st.sidebar.button("Update Frequency"):
    G.nodes[node_to_edit]['frequency'] = new_freq
    st.sidebar.success(f"Node {node_to_edit} set to frequency {new_freq}")

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
    nx.draw(G, pos, node_color=color_map, with_labels=True, ax=ax)
    labels = {n: f"Node {n}\nFreq {G.nodes[n]['frequency']}" for n in G.nodes}
    # nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)
    # Messing up formaatting

    graph_placeholder.pyplot(fig)
    plt.close(fig)  # ✅ Prevent memory overload

    # Display logs
    log.extend(logs_this_tick)
    log_placeholder.code("\n".join(log[-10:]))  # Last 10 logs

    time.sleep(1)  # Real-time pause
