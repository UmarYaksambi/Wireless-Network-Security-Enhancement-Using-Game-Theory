import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import time
# Removed unused: import math
# Removed unused: from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Advanced Wireless Network Security: Game-Theoretic Approach", layout="wide")

# --- SETUP ---
st.title("🚀 Advanced Wireless Network Security: Game-Theoretic Approach")
st.markdown("""
This simulation explores the strategic interactions between an attacker (jammer) and a defender
in a wireless network using various game-theoretic models. Configure the network and simulation
parameters in the sidebar, then start the simulation to observe how strategies evolve and
impact network performance.
""")

# --- NETWORK PARAMETERS ---
with st.sidebar.expander("Network Configuration", expanded=False):
    NUM_NODES = st.slider("Number of Nodes", min_value=4, max_value=20, value=8, key="num_nodes_slider")
    TOPOLOGY = st.selectbox("Network Topology",
                            ["Random (Erdős–Rényi)", "Small-World", "Scale-Free", "Star", "Ring"],
                            key="topology_select")
    # Ensure frequencies are generated based on the slider value *within* the expander context
    num_freq_slider = st.slider("Number of Frequencies", min_value=2, max_value=10, value=5, key="num_freq_slider")
    # Define frequencies in the global scope based on the slider
    frequencies = list(range(1, num_freq_slider + 1))
    connectivity = st.slider("Network Connectivity / Parameter", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="connectivity_slider",
                             help="For Random: Edge probability. For Small-World: Approx k/N. For Scale-Free: Approx m/3.")


# --- GAME THEORY MODEL SELECTION ---
game_model = st.sidebar.selectbox("Game Theory Model",
                                ["Bayesian Game", "Repeated Game", "Stackelberg Game",
                                 "Coalition Formation", "Q-Learning"],
                                key="game_model_select")

# --- STRATEGY SPACES ---
# Added descriptions for clarity
attacker_strategies = ["broadband", "sweep", "reactive", "targeted", "power_burst", "intelligent"]
defender_strategies = ["hop", "detect_and_switch", "stay", "spread_spectrum", "error_coding", "cooperative"]

strategy_descriptions = {
    "Attacker": {
        "broadband": "Jam all frequencies (high cost, high detection).",
        "sweep": "Jam a random subset of frequencies.",
        "reactive": "Jam currently active frequencies.",
        "targeted": "Jam frequencies used by important nodes.",
        "power_burst": "Short, powerful jamming on one frequency (high detection).",
        "intelligent": "Adaptive jamming, attempts to predict/counter defender."
    },
    "Defender": {
        "hop": "Regularly change frequency (medium cost).",
        "detect_and_switch": "Change frequency only when jammed (low cost).",
        "stay": "Remain on the current frequency (very low cost, vulnerable).",
        "spread_spectrum": "Use techniques to resist narrowband jamming (high cost).",
        "error_coding": "Use coding to correct errors from interference (medium cost).",
        "cooperative": "Nodes coordinate to find interference-free channels (low cost per node, needs coordination)."
    }
}


# --- SIMULATION SETTINGS ---
with st.sidebar.expander("Simulation Settings", expanded=True):
    run_sim = st.button("Start Simulation")
    simulation_steps = st.slider("Simulation Steps", min_value=10, max_value=100, value=30, key="sim_steps_slider")
    simulation_speed = st.slider("Simulation Speed (Steps/sec)", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="sim_speed_slider")
    reset_learning = st.checkbox("Reset Learning Models on Start", value=False, help="Check this to clear Q-tables/beliefs when starting.")

# --- ENERGY AND COST MODELS ---
attack_cost = {
    "broadband": 5.0,  # Highest energy consumption
    "sweep": 3.0,      # Medium-high energy
    "reactive": 2.5,   # Medium energy
    "targeted": 2.0,   # Medium-low energy
    "power_burst": 4.0, # High energy but focused
    "intelligent": 1.5 # Lowest energy (efficient)
}

defense_cost = {
    "hop": 2.0,             # Medium energy
    "detect_and_switch": 1.5, # Medium-low energy
    "stay": 0.2,            # Almost no energy
    "spread_spectrum": 3.0, # Higher energy
    "error_coding": 2.5,    # Medium-high energy
    "cooperative": 1.0      # Low energy per node but requires coordination
}

# --- SCORING RULES SIDEBAR ---
with st.sidebar.expander("Scoring Rules", expanded=False):
    st.markdown("""
    #### 📊 Scoring Rules

    Scores reflect the effectiveness and efficiency of strategies.

    ##### Attacker Score:
    * **Base Points**: +2 points per important node jammed.
    * **Strategy Cost**: Penalty based on energy used (see Costs).
    * **Detection Penalty**: -3 points if detected (stochastic based on probability).

    ##### Defender Score:
    * **Base Points**: +2 points per important node protected/operating.
    * **Network Throughput**: +5 points weighted by network connectivity ratio.
    * **Strategy Cost**: Penalty based on energy used (see Costs).
    * *(Implicit)* **Recovery Speed**: Faster recovery (e.g., detect & switch) leads to higher scores over time by minimizing downtime.
    """)

# --- NETWORK GENERATION FUNCTION ---
# Accepts frequencies_list as an argument
def generate_network(topology_type, n_nodes, connect_param, frequencies_list, seed=None):
    """Generates the network graph with specified topology and node properties."""
    # Use a fixed seed for reproducibility unless None
    if seed is None:
        seed = random.randint(0, 10000)

    if topology_type == "Random (Erdős–Rényi)":
        G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)
    elif topology_type == "Small-World":
        k = max(2, int(connect_param * n_nodes))
        if k % 2 != 0: k += 1
        k = min(k, n_nodes - 1)
        p_rewire = 0.3
        if n_nodes <= k:
             st.warning(f"Cannot generate Small-World: N ({n_nodes}) must be > K ({k}). Using Random graph instead.")
             G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)
        else:
             G = nx.watts_strogatz_graph(n_nodes, k, p_rewire, seed=seed)
    elif topology_type == "Scale-Free":
        m = max(1, int(connect_param * 3))
        m = min(m, n_nodes -1)
        if n_nodes <= m:
             st.warning(f"Cannot generate Scale-Free: N ({n_nodes}) must be > M ({m}). Using Random graph instead.")
             G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)
        else:
             G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    elif topology_type == "Star":
        if n_nodes < 2: n_nodes = 2
        G = nx.star_graph(n_nodes - 1)
    elif topology_type == "Ring":
        if n_nodes < 3: n_nodes = 3
        G = nx.cycle_graph(n_nodes)
    else: # Fallback
        G = nx.erdos_renyi_graph(n_nodes, connect_param, seed=seed)

    # Ensure the graph is connected
    if not nx.is_connected(G) and len(G.nodes) > 1:
        components = list(nx.connected_components(G))
        if len(components) > 1:
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i+1]))
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2)

    # Assign node properties using the passed frequencies_list
    if not frequencies_list: frequencies_list = [1, 2] # Safeguard if an empty list is somehow passed
    for node in G.nodes:
        G.nodes[node]['frequency'] = random.choice(frequencies_list) # Use the argument
        G.nodes[node]['status'] = 'idle'
        G.nodes[node]['importance'] = random.uniform(0.5, 1.5)
        G.nodes[node]['power'] = random.uniform(0.7, 1.0)
        G.nodes[node]['SNR'] = random.uniform(15, 30)
        G.nodes[node]['last_freq'] = G.nodes[node]['frequency']

    return G


# --- UTILITY FUNCTION ---
def payoff(attacker_strat, defender_strat, network, success_count, jammed_nodes_list, protected_nodes_list, detection_prob):
    """Calculate complex payoffs based on strategy outcomes."""
    num_total_nodes = len(network.nodes)
    if num_total_nodes == 0: return 0, 0, 0, 0

    jammed_importance_sum = sum(network.nodes[n]['importance'] for n in jammed_nodes_list)
    protected_importance_sum = sum(network.nodes[n]['importance'] for n in protected_nodes_list if n not in jammed_nodes_list)

    operational_nodes = [n for n in network.nodes if network.nodes[n]['status'] != 'jammed']
    connectivity_ratio = 0
    if operational_nodes:
        subgraph = network.subgraph(operational_nodes)
        if subgraph.nodes:
             largest_cc = max(nx.connected_components(subgraph), key=len, default=[])
             connectivity_ratio = len(largest_cc) / num_total_nodes

    attack_effectiveness = jammed_importance_sum * 2
    strategy_cost_atk = attack_cost.get(attacker_strat, 2)
    # Detection penalty is stochastic: penalty applied only if detection "occurs"
    detected = random.random() < detection_prob
    detection_penalty = 3.0 if detected else 0.0
    attacker_payoff = attack_effectiveness - strategy_cost_atk - detection_penalty

    defense_effectiveness = protected_importance_sum * 2
    strategy_cost_def = defense_cost.get(defender_strat, 1)
    network_utility = connectivity_ratio * 5

    if defender_strat == 'cooperative':
         strategy_cost_def = strategy_cost_def * num_total_nodes # Conceptual cost increase

    defender_payoff = defense_effectiveness - strategy_cost_def + network_utility

    return strategy_cost_atk, strategy_cost_def, attacker_payoff, defender_payoff, detected # Return detection status


# --- ADVANCED ATTACK IMPLEMENTATION ---
# Accepts frequencies_list as an argument
def execute_attack(strategy, network, defender_strategy, frequencies_list):
    """Implement different attack strategies with varying effectiveness"""
    jammed_freqs = set()
    targeted_nodes_ids = []
    detection_probability = 0.0
    num_nodes = len(network.nodes)
    if num_nodes == 0: return set(), [], 0.0
    if not frequencies_list: return set(), [], 0.0 # Check passed argument

    active_freqs = {network.nodes[n]['frequency'] for n in network.nodes}
    nodes_by_importance = sorted(network.nodes,
                                 key=lambda n: network.nodes[n]['importance'],
                                 reverse=True)

    if strategy == "broadband":
        jammed_freqs.update(frequencies_list) # Use argument
        detection_probability = 0.8

    elif strategy == "sweep":
        n_freqs_to_jam = min(len(frequencies_list), max(1, len(frequencies_list) // 2))
        if len(frequencies_list) > 0:
             jammed_freqs.update(random.sample(frequencies_list, n_freqs_to_jam)) # Use argument
        detection_probability = 0.5

    elif strategy == "reactive":
        if active_freqs:
            jammed_freqs.add(random.choice(list(active_freqs)))
        detection_probability = 0.3

    elif strategy == "targeted":
        priority_nodes = nodes_by_importance[:max(1, num_nodes // 3)]
        target_freqs = {network.nodes[n]['frequency'] for n in priority_nodes}
        jammed_freqs.update(target_freqs)
        targeted_nodes_ids = priority_nodes
        detection_probability = 0.4

    elif strategy == "power_burst":
        if frequencies_list:
             jammed_freqs.add(random.choice(frequencies_list)) # Use argument
        detection_probability = 0.7

    elif strategy == "intelligent":
        detection_probability = 0.15
        if defender_strategy == "hop":
            prediction_success_prob = 0.4
            if random.random() < prediction_success_prob:
                 possible_next = [f for f in frequencies_list if f not in active_freqs] # Use argument
                 if not possible_next: possible_next = frequencies_list # Use argument
                 jammed_freqs.update(random.sample(possible_next, min(2, len(possible_next))))
                 detection_probability = 0.2
            else:
                 if frequencies_list: # Use argument
                      jammed_freqs.add(random.choice(frequencies_list)) # Use argument
        elif defender_strategy == "stay":
             jammed_freqs.update(active_freqs)
             detection_probability = 0.3
        elif defender_strategy == "detect_and_switch":
             if active_freqs: # Check if active_freqs is not empty
                 jammed_freqs.update(random.sample(list(active_freqs), min(len(active_freqs), 2)))
             detection_probability = 0.25
        else:
            if active_freqs:
                jammed_freqs.add(random.choice(list(active_freqs)))

    if defender_strategy in ["spread_spectrum", "error_coding", "cooperative"]:
        detection_probability = min(1.0, detection_probability + 0.15)

    return jammed_freqs, targeted_nodes_ids, detection_probability


# --- ADVANCED DEFENSE IMPLEMENTATION ---
# Accepts frequencies_list as an argument
def execute_defense(strategy, network, jammed_freqs_set, frequencies_list):
    """Implement different defense strategies with varying effectiveness"""
    if not frequencies_list: return network # Check passed argument

    for node in network.nodes:
        current_freq = network.nodes[node]['frequency']
        network.nodes[node]['last_freq'] = current_freq
        node_status = 'unknown' # Represents action, not final state
        is_jammed = current_freq in jammed_freqs_set

        if strategy == "hop":
            available_freqs = [f for f in frequencies_list if f != current_freq] # Use argument
            if not available_freqs: available_freqs = frequencies_list # Use argument
            if available_freqs:
                 network.nodes[node]['frequency'] = random.choice(available_freqs)
            node_status = 'hopping'

        elif strategy == "detect_and_switch":
            if is_jammed:
                available_freqs = [f for f in frequencies_list if f != current_freq] # Use argument
                if not available_freqs: available_freqs = frequencies_list # Use argument
                if available_freqs:
                    network.nodes[node]['frequency'] = random.choice(available_freqs)
                node_status = 'switching'
            else:
                node_status = 'staying'

        elif strategy == "stay":
             node_status = 'staying'

        elif strategy == "spread_spectrum":
            node_status = 'spread_spectrum_active'

        elif strategy == "error_coding":
             node_status = 'error_coding_active'

        elif strategy == "cooperative":
            neighbors = list(network.neighbors(node))
            neighbor_freqs = {network.nodes[n]['frequency'] for n in neighbors}
            preferred_available = [f for f in frequencies_list if f not in neighbor_freqs and f not in jammed_freqs_set] # Use argument

            if preferred_available:
                 network.nodes[node]['frequency'] = random.choice(preferred_available)
                 node_status = 'coordinating_switch'
            else:
                 non_jammed_available = [f for f in frequencies_list if f not in jammed_freqs_set] # Use argument
                 if non_jammed_available:
                      network.nodes[node]['frequency'] = random.choice(non_jammed_available)
                      node_status = 'coordinating_switch'
                 else:
                      available_freqs = [f for f in frequencies_list if f != current_freq] # Use argument
                      if available_freqs:
                           network.nodes[node]['frequency'] = random.choice(available_freqs)
                           node_status = 'coordinating_fallback_hop'
                      else:
                           node_status = 'coordinating_failed_stay'
        # Store the action taken for potential logging/analysis
        network.nodes[node]['defense_action'] = node_status

    return network


# --- LEARNING PARAMETERS & STATE INIT ---
def initialize_learning_models(force_reset=False):
    """Initializes or resets Q-tables and Bayesian beliefs in session state."""
    global alpha, gamma, epsilon # Allow modification if needed later

    if game_model == "Q-Learning":
        if "q_attacker" not in st.session_state or force_reset:
            st.session_state.q_attacker = {a: {d: 0.0 for d in defender_strategies} for a in attacker_strategies}
            st.session_state.q_defender = {d: {a: 0.0 for a in attacker_strategies} for d in defender_strategies}
            st.session_state.q_learning_initialized = True
            if force_reset: print("Q-Tables Initialized/Reset")

    if game_model == "Bayesian Game":
        if "attacker_belief" not in st.session_state or force_reset:
            st.session_state.attacker_belief = {d: 1.0 / len(defender_strategies) for d in defender_strategies}
            st.session_state.defender_belief = {a: 1.0 / len(attacker_strategies) for a in attacker_strategies}
            st.session_state.attacker_success_rate = {a: 0.5 for a in attacker_strategies}
            st.session_state.defender_success_rate = {d: 0.5 for d in defender_strategies}
            # Initialize average payoff trackers
            st.session_state.avg_a_payoff = 0.0
            st.session_state.avg_d_payoff = 0.0
            st.session_state.avg_payoffs_atk = {} # For (a,d) pairs
            st.session_state.avg_payoffs_def = {} # For (d,a) pairs
            st.session_state.bayesian_initialized = True
            if force_reset: print("Bayesian Beliefs Initialized/Reset")

# Learning parameters
alpha = 0.1  # Learning rate (Q-Learning, Bayesian update decay)
gamma = 0.9  # Discount factor (Q-Learning)
epsilon = 0.2  # Exploration rate (Q-Learning)

# Call initialization once at the start if session state isn't set up
if "session_initialized" not in st.session_state:
     initialize_learning_models(force_reset=True)
     st.session_state.session_initialized = True


# --- COALITION FORMATION ---
def assign_coalitions(network):
     """Assigns nodes to random coalitions."""
     num_coalitions = max(1, min(3, len(network.nodes) // 3))
     for node in network.nodes:
         network.nodes[node]['coalition'] = random.randint(0, num_coalitions - 1)
     return network


# --- STATE TRACKERS (Ensure initialization) ---
if "a_score" not in st.session_state: st.session_state.a_score = 0.0
if "d_score" not in st.session_state: st.session_state.d_score = 0.0
if "history" not in st.session_state:
    st.session_state.history = {
        "step": [], "attacker_strategies": [], "defender_strategies": [],
        "attacker_payoffs": [], "defender_payoffs": [],
        "attacker_costs": [], "defender_costs": [],
        "jammed_nodes_count": [], "network_health": [],
        "attacker_detected": []
    }

# --- UI PLACEHOLDERS ---
col_main, col_sidebar_sim = st.columns([3, 1])

with col_main:
    graph_placeholder = st.empty()
    historical_placeholder = st.empty()

with col_sidebar_sim:
    metrics_placeholder = st.empty() 
    log_placeholder = st.expander("Simulation Log", expanded=False)

# --- STRATEGY SELECTION BASED ON GAME MODEL ---
def select_strategies(game_type, step, current_network):
    """Select strategies based on the chosen game theory model"""
    atk = random.choice(attacker_strategies)
    dfd = random.choice(defender_strategies)

    history_len = len(st.session_state.history.get("step", []))

    if game_type == "Bayesian Game":
        if "attacker_belief" not in st.session_state: initialize_learning_models(force_reset=True)

        def get_best_response_heuristic(my_strategies, opponent_strategies, opponent_belief, payoff_estimator):
             expected_payoffs = {}
             for my_strat in my_strategies:
                  exp_payoff = sum(opponent_belief.get(opp_strat, 0) *
                                   payoff_estimator(my_strat, opp_strat)
                                   for opp_strat in opponent_strategies)
                  expected_payoffs[my_strat] = exp_payoff
             if expected_payoffs:
                  # Add small random noise to break ties and encourage exploration
                  noisy_payoffs = {s: p + random.gauss(0, 0.01) for s, p in expected_payoffs.items()}
                  best_strat = max(noisy_payoffs, key=noisy_payoffs.get)
                  return best_strat
             else: return random.choice(my_strategies)

        def estimate_attacker_payoff(a_strat, d_strat):
             key = (a_strat, d_strat)
             return st.session_state.get("avg_payoffs_atk", {}).get(key, 0) # Use stored average

        def estimate_defender_payoff(d_strat, a_strat):
             key = (d_strat, a_strat)
             return st.session_state.get("avg_payoffs_def", {}).get(key, 0) # Use stored average

        if random.random() < 0.8: # Exploit probability
            atk = get_best_response_heuristic(attacker_strategies, defender_strategies, st.session_state.defender_belief, estimate_attacker_payoff)
            dfd = get_best_response_heuristic(defender_strategies, attacker_strategies, st.session_state.attacker_belief, estimate_defender_payoff)
        else: # Explore
            atk = random.choice(attacker_strategies)
            dfd = random.choice(defender_strategies)

    elif game_type == "Repeated Game":
        if history_len > 5:
            lookback = 5
            # Check if history keys exist before accessing
            recent_def_strats = st.session_state.history.get("defender_strategies", [])[-lookback:]
            recent_atk_strats = st.session_state.history.get("attacker_strategies", [])[-lookback:]
            recent_def_payoffs = st.session_state.history.get("defender_payoffs", [])[-lookback:]
            recent_atk_payoffs = st.session_state.history.get("attacker_payoffs", [])[-lookback:]

            # Ensure history is long enough
            if len(recent_def_strats) == lookback:
                atk_effectiveness = {a: 0.0 for a in attacker_strategies}
                atk_counts = {a: 0 for a in attacker_strategies}
                for i in range(lookback):
                     a_strat = recent_atk_strats[i]
                     atk_effectiveness[a_strat] += recent_atk_payoffs[i]
                     atk_counts[a_strat] += 1
                avg_atk_effectiveness = {a: atk_effectiveness[a] / atk_counts[a] if atk_counts[a] > 0 else -float('inf') for a in attacker_strategies}

                def_effectiveness = {d: 0.0 for d in defender_strategies}
                def_counts = {d: 0 for d in defender_strategies}
                for i in range(lookback):
                     d_strat = recent_def_strats[i]
                     def_effectiveness[d_strat] += recent_def_payoffs[i]
                     def_counts[d_strat] += 1
                avg_def_effectiveness = {d: def_effectiveness[d] / def_counts[d] if def_counts[d] > 0 else -float('inf') for d in defender_strategies}

                if random.random() < 0.7 and any(p > -float('inf') for p in avg_atk_effectiveness.values()):
                    atk = max(avg_atk_effectiveness, key=avg_atk_effectiveness.get)
                else: atk = random.choice(attacker_strategies)

                if random.random() < 0.7 and any(p > -float('inf') for p in avg_def_effectiveness.values()):
                    dfd = max(avg_def_effectiveness, key=avg_def_effectiveness.get)
                else: dfd = random.choice(defender_strategies)
            # else: initial random choice remains

    elif game_type == "Stackelberg Game":
        # Defender Leader chooses
        if history_len > 3 and random.random() < 0.7:
            def_effectiveness = {d: 0.0 for d in defender_strategies}
            def_counts = {d: 0 for d in defender_strategies}
            payoffs = st.session_state.history.get("defender_payoffs", [])
            strats = st.session_state.history.get("defender_strategies", [])
            for i in range(len(payoffs)):
                 d_strat = strats[i]
                 def_effectiveness[d_strat] += payoffs[i]
                 def_counts[d_strat] += 1
            avg_def_effectiveness = {d: def_effectiveness[d] / def_counts[d] if def_counts[d] > 0 else -float('inf') for d in defender_strategies}
            if any(p > -float('inf') for p in avg_def_effectiveness.values()):
                dfd = max(avg_def_effectiveness, key=avg_def_effectiveness.get)
            else: dfd = random.choice(defender_strategies)
        else: dfd = random.choice(defender_strategies)

        # Attacker Follower responds
        if history_len > 3:
             response_payoffs = {a: 0.0 for a in attacker_strategies}
             response_counts = {a: 0 for a in attacker_strategies}
             atk_payoffs = st.session_state.history.get("attacker_payoffs", [])
             atk_strats = st.session_state.history.get("attacker_strategies", [])
             def_strats = st.session_state.history.get("defender_strategies", [])
             for i in range(len(atk_payoffs)):
                  if def_strats[i] == dfd: # If defender played the chosen strategy `dfd`
                       a_strat_past = atk_strats[i]
                       response_payoffs[a_strat_past] += atk_payoffs[i]
                       response_counts[a_strat_past] += 1
             avg_response_payoffs = {a: response_payoffs[a] / response_counts[a] if response_counts[a] > 0 else -float('inf') for a in attacker_strategies}

             if random.random() < 0.8 and any(p > -float('inf') for p in avg_response_payoffs.values()):
                  atk = max(avg_response_payoffs, key=avg_response_payoffs.get)
             else: atk = random.choice(attacker_strategies)
        else: atk = random.choice(attacker_strategies)

    elif game_type == "Coalition Formation":
        # Ensure coalitions are assigned
        if not current_network.nodes or 'coalition' not in current_network.nodes[list(current_network.nodes)[0]]:
             current_network = assign_coalitions(current_network)

        coalition_strengths = {}
        unique_coalitions = set(nx.get_node_attributes(current_network, 'coalition').values())
        if unique_coalitions: # Check if coalitions exist
            for c_id in unique_coalitions:
                coalition_nodes = [n for n, data in current_network.nodes(data=True) if data.get('coalition') == c_id]
                if coalition_nodes:
                    avg_importance = sum(current_network.nodes[n]['importance'] for n in coalition_nodes) / len(coalition_nodes)
                    coalition_strengths[c_id] = len(coalition_nodes) * avg_importance

            if coalition_strengths: # Check if strengths calculated
                total_strength = sum(coalition_strengths.values())
                strongest_coalition_id = max(coalition_strengths, key=coalition_strengths.get, default=None)
                if strongest_coalition_id is not None and coalition_strengths[strongest_coalition_id] > 0.6 * total_strength:
                     dfd = "cooperative"
                else: dfd = random.choice(["hop", "detect_and_switch", "error_coding"])
            else: dfd = random.choice(defender_strategies) # Fallback if no coalitions/strengths

            if coalition_strengths and random.random() < 0.6:
                 atk = "targeted" # Attacker targets strongest coalition (logic in execute_attack needs adaptation)
            else: atk = random.choice(["sweep", "reactive", "broadband"])
        else: # Fallback if no nodes or coalitions
            dfd = random.choice(defender_strategies)
            atk = random.choice(attacker_strategies)


    elif game_type == "Q-Learning":
        if "q_attacker" not in st.session_state: initialize_learning_models(force_reset=True)

        if random.random() < epsilon:
            atk = random.choice(attacker_strategies)
            dfd = random.choice(defender_strategies)
        else:
            # Exploit - Attacker selects based on Defender's last known action (state)
            if history_len > 0 and "defender_strategies" in st.session_state.history:
                last_defender_action = st.session_state.history["defender_strategies"][-1]
                q_values_atk = {a: st.session_state.q_attacker[a].get(last_defender_action, 0) for a in attacker_strategies}
                if q_values_atk: atk = max(q_values_atk, key=q_values_atk.get)
                else: atk = random.choice(attacker_strategies)
            else: atk = random.choice(attacker_strategies) # No history state

            # Exploit - Defender selects based on Attacker's last known action (state)
            if history_len > 0 and "attacker_strategies" in st.session_state.history:
                last_attacker_action = st.session_state.history["attacker_strategies"][-1]
                q_values_def = {d: st.session_state.q_defender[d].get(last_attacker_action, 0) for d in defender_strategies}
                if q_values_def: dfd = max(q_values_def, key=q_values_def.get)
                else: dfd = random.choice(defender_strategies)
            else: dfd = random.choice(defender_strategies) # No history state

    return atk, dfd


# --- UPDATE LEARNING MODELS ---
def update_learning_models(game_type, atk, dfd, a_payoff, d_payoff):
    """Update learning models based on game outcome"""
    history_len = len(st.session_state.history.get("step", []))

    if game_type == "Bayesian Game":
        if "attacker_belief" not in st.session_state: initialize_learning_models(force_reset=True)
        decay = 1.0 - alpha

        # Update success rates heuristically
        atk_success = a_payoff > st.session_state.get("avg_a_payoff", 0)
        def_success = d_payoff > st.session_state.get("avg_d_payoff", 0)
        st.session_state.attacker_success_rate[atk] = (decay * st.session_state.attacker_success_rate.get(atk, 0.5) + alpha * float(atk_success))
        st.session_state.defender_success_rate[dfd] = (decay * st.session_state.defender_success_rate.get(dfd, 0.5) + alpha * float(def_success))

        # Update beliefs (normalized success rates)
        total_atk_success = sum(st.session_state.attacker_success_rate.values())
        if total_atk_success > 0:
            st.session_state.defender_belief = {a: st.session_state.attacker_success_rate[a] / total_atk_success for a in attacker_strategies}
        else: st.session_state.defender_belief = {a: 1.0 / len(attacker_strategies) for a in attacker_strategies}

        total_def_success = sum(st.session_state.defender_success_rate.values())
        if total_def_success > 0:
            st.session_state.attacker_belief = {d: st.session_state.defender_success_rate[d] / total_def_success for d in defender_strategies}
        else: st.session_state.attacker_belief = {d: 1.0 / len(defender_strategies) for d in defender_strategies}

        # Update global average payoffs and strategy-pair average payoffs
        st.session_state.avg_a_payoff = (decay * st.session_state.get("avg_a_payoff", 0) + alpha * a_payoff)
        st.session_state.avg_d_payoff = (decay * st.session_state.get("avg_d_payoff", 0) + alpha * d_payoff)
        key_atk = (atk, dfd)
        key_def = (dfd, atk)
        st.session_state.avg_payoffs_atk[key_atk] = (decay * st.session_state.avg_payoffs_atk.get(key_atk, 0) + alpha * a_payoff)
        st.session_state.avg_payoffs_def[key_def] = (decay * st.session_state.avg_payoffs_def.get(key_def, 0) + alpha * d_payoff)


    elif game_type == "Q-Learning":
        if "q_attacker" not in st.session_state: initialize_learning_models(force_reset=True)
        # Q-value update for the actions *just taken* (atk, dfd)
        # Assumes state is implicitly opponent's action in this round for next step prediction

        # Predict max Q for the next state (based on current opponent actions)
        max_future_q_atk = max(st.session_state.q_attacker[next_a].get(dfd, 0) for next_a in attacker_strategies)
        max_future_q_def = max(st.session_state.q_defender[next_d].get(atk, 0) for next_d in defender_strategies)

        # Get current Q-values for the actions taken
        current_q_atk = st.session_state.q_attacker[atk].get(dfd, 0)
        current_q_def = st.session_state.q_defender[dfd].get(atk, 0)

        # Update Q-values using the Bellman equation
        st.session_state.q_attacker[atk][dfd] = current_q_atk + alpha * (a_payoff + gamma * max_future_q_atk - current_q_atk)
        st.session_state.q_defender[dfd][atk] = current_q_def + alpha * (d_payoff + gamma * max_future_q_def - current_q_def)


# --- SIMULATION LOOP ---
if run_sim:
    # --- Reset State for new simulation ---
    st.session_state.a_score = 0.0
    st.session_state.d_score = 0.0
    st.session_state.history = {
        "step": [], "attacker_strategies": [], "defender_strategies": [],
        "attacker_payoffs": [], "defender_payoffs": [],
        "attacker_costs": [], "defender_costs": [],
        "jammed_nodes_count": [], "network_health": [],
        "attacker_detected": []
    }
    if reset_learning:
        initialize_learning_models(force_reset=True)
        st.info("Learning models reset.")
    else:
        # Ensure models are initialized if they weren't already
        initialize_learning_models(force_reset=False)


    # --- Generate Network based on current settings ---
    # Pass the global frequencies list defined from the slider
    current_G = generate_network(TOPOLOGY, NUM_NODES, connectivity, frequencies, seed=42)
    if game_model == "Coalition Formation":
        current_G = assign_coalitions(current_G)


    logs = ["Simulation Started..."]
    log_placeholder.text_area("Log", "".join(logs), height=200, key="log_init")

    progress_bar = st.progress(0)

    for step in range(simulation_steps):
        step_logs = [f"\n------ Step {step+1} / {simulation_steps} ------"]

        # --- STRATEGY SELECTION ---
        atk, dfd = select_strategies(game_model, step, current_G)
        step_logs.append(f"🔥 Attacker Strategy: {atk} ({strategy_descriptions['Attacker'].get(atk, '')})")
        step_logs.append(f"🛡️ Defender Strategy: {dfd} ({strategy_descriptions['Defender'].get(dfd, '')})")

        # --- ATTACK EXECUTION ---
        # Pass the global frequencies list
        jammed_freqs_set, targeted_nodes_ids, detection_prob = execute_attack(atk, current_G, dfd, frequencies)
        step_logs.append(f"📡 Attacker jamming frequencies: {jammed_freqs_set if jammed_freqs_set else 'None'}")
        if targeted_nodes_ids: step_logs.append(f"🎯 Attacker targeted nodes: {targeted_nodes_ids}")
        step_logs.append(f"🕵️ Attacker base detection probability: {detection_prob:.2f}")

        # --- DEFENSE EXECUTION ---
        # Pass the global frequencies list
        current_G = execute_defense(dfd, current_G, jammed_freqs_set, frequencies)

        # --- OUTCOME EVALUATION ---
        success_count = 0
        jammed_nodes_list = []
        protected_nodes_list = []
        resistant_nodes_list = []

        for node in current_G.nodes:
            final_freq = current_G.nodes[node]['frequency']
            is_on_jammed_freq = final_freq in jammed_freqs_set
            current_G.nodes[node]['status'] = 'idle' # Reset status

            if is_on_jammed_freq:
                node_can_resist = False
                 # Use global `frequencies` length for total spectrum size comparison
                if dfd == "spread_spectrum" and len(jammed_freqs_set) < len(frequencies) * 0.6:
                     node_can_resist = True
                elif dfd == "error_coding" and random.random() < 0.6:
                     node_can_resist = True

                if node_can_resist:
                    current_G.nodes[node]['status'] = 'resistant'
                    protected_nodes_list.append(node)
                    resistant_nodes_list.append(node)
                    success_count += 1
                else:
                    current_G.nodes[node]['status'] = 'jammed'
                    jammed_nodes_list.append(node)
            else:
                current_G.nodes[node]['status'] = 'ok'
                protected_nodes_list.append(node)
                success_count += 1

        jammed_count = len(jammed_nodes_list)
        network_health = (NUM_NODES - jammed_count) / NUM_NODES if NUM_NODES > 0 else 0
        step_logs.append(f"📊 Outcome: {success_count} OK/Resistant, {jammed_count} Jammed. Health: {network_health:.1%}")
        if resistant_nodes_list: step_logs.append(f"💪 Resistant Nodes: {resistant_nodes_list}")

        # --- SCORING ---
        atk_cost_val, def_cost_val, a_reward, d_reward, detected_this_step = payoff(
            atk, dfd, current_G, success_count, jammed_nodes_list, protected_nodes_list, detection_prob
        )
        st.session_state.a_score += a_reward
        st.session_state.d_score += d_reward
        step_logs.append(f"💰 Payoffs: Attacker={a_reward:.2f} (Cost:{atk_cost_val:.1f}), Defender={d_reward:.2f} (Cost:{def_cost_val:.1f})")
        if detected_this_step: step_logs.append("🚨 Attacker Detected!")
        step_logs.append(f"📈 Cumulative Scores: Attacker={st.session_state.a_score:.2f}, Defender={st.session_state.d_score:.2f}")

        # --- UPDATE HISTORY ---
        st.session_state.history["step"].append(step + 1)
        st.session_state.history["attacker_strategies"].append(atk)
        st.session_state.history["defender_strategies"].append(dfd)
        st.session_state.history["attacker_payoffs"].append(a_reward)
        st.session_state.history["defender_payoffs"].append(d_reward)
        st.session_state.history["attacker_costs"].append(atk_cost_val)
        st.session_state.history["defender_costs"].append(def_cost_val)
        st.session_state.history["jammed_nodes_count"].append(jammed_count)
        st.session_state.history["network_health"].append(network_health)
        st.session_state.history["attacker_detected"].append(detected_this_step)

        # --- UPDATE LEARNING MODELS ---
        update_learning_models(game_model, atk, dfd, a_reward, d_reward)

        # --- VISUALIZATION ---
        fig_graph, ax_graph = plt.subplots(figsize=(8, 6))
        color_map = []
        status_colors = {'ok': 'green', 'jammed': 'red', 'resistant': 'orange', 'idle':'grey'}
        for n in current_G.nodes:
            color_map.append(status_colors.get(current_G.nodes[n]['status'], 'grey'))
        node_sizes = [200 + 400 * current_G.nodes[n]['importance'] for n in current_G.nodes]
        try:
             pos = nx.spring_layout(current_G, seed=42, k=0.8/math.sqrt(len(current_G.nodes)) if len(current_G.nodes)>0 else 0.5) # Adjust k for spacing
        except Exception: # Catch potential layout errors
             pos = nx.random_layout(current_G, seed=42)

        nx.draw(current_G, pos, node_color=color_map, with_labels=True, ax=ax_graph,
                node_size=node_sizes, font_size=9, font_weight='bold', edge_color='gray', width=0.5)
        freq_labels = {n: f"F:{current_G.nodes[n]['frequency']}" for n in current_G.nodes}
        label_pos = {k: [v[0], v[1]-0.12] for k,v in pos.items()} # Adjusted offset
        nx.draw_networkx_labels(current_G, label_pos, labels=freq_labels, font_size=8, ax=ax_graph, font_color='purple')
        ax_graph.set_title(f"Network Status (Step {step+1}) - {TOPOLOGY}")
        ax_graph.set_xticks([])
        ax_graph.set_yticks([])
        plt.tight_layout() # Adjust layout

        with graph_placeholder.container():
             st.pyplot(fig_graph)
        plt.close(fig_graph)

        # --- METRICS DISPLAY ---
        with metrics_placeholder.container():
            st.markdown("---")
            st.markdown(f"#### 📊 Metrics at Step {step+1}")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Defender Score", f"{st.session_state.d_score:.2f}", delta=f"{d_reward:.2f}")
                st.metric("Attacker Score", f"{st.session_state.a_score:.2f}", delta=f"{a_reward:.2f}")
            with m_col2:
                st.metric("Network Health", f"{network_health:.1%}")
                st.metric("Jammed Nodes", f"{jammed_count}/{NUM_NODES}")

            st.markdown(f"**Defender Strategy:** `{dfd}`")
            st.markdown(f"**Attacker Strategy:** `{atk}`")
            if detected_this_step: st.error("🚨 Attacker Detected!")
            # Update progress bar here inside the container update
            progress_bar.progress((step + 1) / simulation_steps)


        # --- LOG DISPLAY ---
        logs.extend(step_logs)
        max_log_lines = 150
        display_logs = logs[-max_log_lines:]
        log_placeholder.text_area("Log", "\n".join(reversed(display_logs)), height=300, key=f"log_{step}")


        # --- PAUSE ---
        time.sleep(1.0 / simulation_speed)

    # Final log update after loop
    logs.append("\n------ Simulation Finished ------")
    log_placeholder.text_area("Log", "\n".join(reversed(logs[-max_log_lines:])), height=300, key="log_final")
    # Ensure progress bar shows 100%
    progress_bar.progress(1.0)


# --- POST-SIMULATION ANALYSIS ---
if len(st.session_state.history["step"]) > 0:
    st.toast("Simulation finished!")
    with historical_placeholder.container():
        st.markdown("---")
        st.markdown("### 📈 Simulation History & Analysis")

        hist_df = pd.DataFrame(st.session_state.history)

        # --- Payoff History Plot ---
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        hist_df.plot(x='step', y=['attacker_payoffs', 'defender_payoffs'], ax=ax_hist,
                     label=['Attacker Payoff', 'Defender Payoff'], marker='.', linestyle='-', markersize=5)
        ax_hist.set_title("Payoff History Over Simulation Steps")
        ax_hist.set_xlabel("Step")
        ax_hist.set_ylabel("Payoff")
        ax_hist.grid(True, linestyle='--', alpha=0.6)
        ax_hist.legend()
        ax_hist.axhline(0, color='black', linewidth=0.5, linestyle='--')
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # --- Network Health History Plot ---
        fig_health, ax_health = plt.subplots(figsize=(10, 4))
        hist_df['network_health_percent'] = hist_df['network_health'] * 100 # Convert to percentage
        hist_df.plot(x='step', y='network_health_percent', ax=ax_health,
                     label='Network Health (%)', color='teal', marker='.', linestyle='-', markersize=5)
        ax_health.set_title("Network Health Over Simulation Steps")
        ax_health.set_xlabel("Step")
        ax_health.set_ylabel("Health (%)")
        ax_health.set_ylim(0, 105) # Set y-axis limit
        ax_health.grid(True, linestyle='--', alpha=0.6)
        ax_health.legend()
        st.pyplot(fig_health)
        plt.close(fig_health)


        # --- Strategy Analysis ---
        if len(st.session_state.history["step"]) > 5:
            st.markdown("#### 🔍 Strategy Analysis")
            analysis_col1, analysis_col2 = st.columns(2)

            with analysis_col1:
                st.subheader("Strategy Usage Frequency")
                fig_freq, (ax_freq1, ax_freq2) = plt.subplots(2, 1, figsize=(6, 7), sharex=False)
                try:
                    atk_counts = hist_df['attacker_strategies'].value_counts().reindex(attacker_strategies, fill_value=0)
                    def_counts = hist_df['defender_strategies'].value_counts().reindex(defender_strategies, fill_value=0)

                    atk_counts.plot(kind='bar', ax=ax_freq1, color='crimson', rot=30)
                    ax_freq1.set_title("Attacker Strategy Usage")
                    ax_freq1.set_ylabel("Count")
                    ax_freq1.tick_params(axis='x', labelsize=8)
                    ax_freq1.grid(True, axis='y', linestyle='--', alpha=0.5)

                    def_counts.plot(kind='bar', ax=ax_freq2, color='navy', rot=30)
                    ax_freq2.set_title("Defender Strategy Usage")
                    ax_freq2.set_xlabel("Strategy")
                    ax_freq2.set_ylabel("Count")
                    ax_freq2.tick_params(axis='x', labelsize=8)
                    ax_freq2.grid(True, axis='y', linestyle='--', alpha=0.5)

                    fig_freq.tight_layout(pad=0.5) # Adjusted padding
                    st.pyplot(fig_freq)
                except Exception as e:
                    st.error(f"Error plotting frequency analysis: {e}")
                finally:
                    plt.close(fig_freq)

            with analysis_col2:
                st.subheader("Strategy Effectiveness")
                fig_eff, (ax_eff1, ax_eff2) = plt.subplots(2, 1, figsize=(6, 7), sharex=False)
                try:
                    avg_atk_payoffs = hist_df.groupby('attacker_strategies')['attacker_payoffs'].mean().reindex(attacker_strategies, fill_value=0)
                    avg_def_payoffs = hist_df.groupby('defender_strategies')['defender_payoffs'].mean().reindex(defender_strategies, fill_value=0)

                    avg_atk_payoffs.plot(kind='bar', ax=ax_eff1, color='darkred', rot=30)
                    ax_eff1.set_title("Attacker Avg. Payoff per Strategy")
                    ax_eff1.set_ylabel("Average Payoff")
                    ax_eff1.tick_params(axis='x', labelsize=8)
                    ax_eff1.grid(True, axis='y', linestyle='--', alpha=0.5)
                    ax_eff1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

                    avg_def_payoffs.plot(kind='bar', ax=ax_eff2, color='darkblue', rot=30)
                    ax_eff2.set_title("Defender Avg. Payoff per Strategy")
                    ax_eff2.set_xlabel("Strategy")
                    ax_eff2.set_ylabel("Average Payoff")
                    ax_eff2.tick_params(axis='x', labelsize=8)
                    ax_eff2.grid(True, axis='y', linestyle='--', alpha=0.5)
                    ax_eff2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

                    fig_eff.tight_layout(pad=0.5) # Adjusted padding
                    st.pyplot(fig_eff)
                except Exception as e:
                    st.error(f"Error plotting effectiveness analysis: {e}")
                finally:
                    plt.close(fig_eff)

        # --- Display Raw History Data ---
        with st.expander("Show Raw Simulation History Data"):
             # Select subset of columns for better display
             display_cols = ["step", "attacker_strategies", "defender_strategies",
                             "attacker_payoffs", "defender_payoffs", "network_health", "attacker_detected"]
             st.dataframe(hist_df[display_cols].round(2)) # Round floats for display

# --- Display Initial Network if simulation hasn't run ---
elif not run_sim and (not st.session_state.history or not st.session_state.history["step"]):
     with graph_placeholder.container():
         st.markdown("---")
         st.subheader("Initial Network Preview")
         st.markdown("*(Based on current sidebar settings. Press 'Start Simulation' to run)*")
         try:
             # Pass the global frequencies list defined from the slider
             initial_G = generate_network(TOPOLOGY, NUM_NODES, connectivity, frequencies, seed=42)
             fig_init, ax_init = plt.subplots(figsize=(8, 6))
             pos_init = nx.spring_layout(initial_G, seed=42)
             node_colors = [initial_G.nodes[n]['importance'] for n in initial_G.nodes] # Color by importance
             nx.draw(initial_G, pos_init, with_labels=True, ax=ax_init, node_color=node_colors,
                     cmap=plt.cm.viridis, node_size=500, font_size=10)
             ax_init.set_title(f"Initial Network Preview ({TOPOLOGY})")
             st.pyplot(fig_init)
             plt.close(fig_init)
         except Exception as e:
             st.error(f"Could not generate initial network preview: {e}")