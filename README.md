# Wireless Network Security Enhancement Using Game Theory

Welcome to the **Advanced Wireless Network Security Simulation**, a project that models the strategic interaction between attackers (jammers) and defenders in wireless networks using **Game Theory**.  
This simulation helps study how different strategies impact the resilience and performance of a wireless network under attack.

> 🔗 Live Codebase: [NetSim.py](https://github.com/UmarYaksambi/Wireless-Network-Security-Enhancement-Using-Game-Theory/blob/main/NetSim.py)  
> ☕ Love this project? [Buy me a coffee](https://www.buymeacoffee.com/umaryaksambi)!

---

## 📦 Features

- **Dynamic Wireless Networks**  
  - Configurable Topologies: Random, Small-World, Scale-Free, Star, Ring  
  - Node properties: frequency, signal strength, importance
- **Game-Theoretic Models**  
  - Bayesian Game  
  - Repeated Game  
  - Stackelberg Game  
  - Coalition Formation  
  - Q-Learning based Adaptation
- **Attack Strategies**  
  - Broadband, Sweep, Reactive, Targeted, Power Burst, Intelligent Attacks
- **Defense Strategies**  
  - Frequency Hopping, Detection & Switching, Spread Spectrum, Error Coding, Cooperative Defense
- **Real-time Visualization**  
  - Network health, connectivity, resistance, and jamming over time
- **Strategy Performance Analytics**  
  - Payoff graphs, network health history, strategy effectiveness analysis

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/UmarYaksambi/Wireless-Network-Security-Enhancement-Using-Game-Theory.git
cd Wireless-Network-Security-Enhancement-Using-Game-Theory
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` includes:**
- `streamlit`
- `networkx`
- `matplotlib`
- `numpy`
- `pandas`

*(Make sure you have Python 3.8+)*

### 3. Run the Simulation

```bash
streamlit run NetSim.py
```

A local Streamlit app will open in your browser, allowing you to interact with the simulation parameters.

---

## 📂 Project Structure

| File | Description |
|:-----|:------------|
| `NetSim.py` | **Main Simulation**: complete Streamlit app with interactive wireless network security simulation. |
| `network.py` | Preliminary earlier attempt — **basic network generation** (now replaced by better architecture in `NetSim.py`). |
| `requirements.txt` | Python libraries required to run the simulation. |

---

## 🎯 How It Works

- **Users configure** the network (nodes, connectivity, number of frequencies) and **select a Game Theory model**.
- **Attackers** dynamically select strategies to jam nodes.
- **Defenders** employ strategies to adapt, protect, and restore network connectivity.
- **Payoffs** are calculated for both sides, considering cost, effectiveness, and detection.
- **Machine Learning / Reinforcement Learning** (Q-Learning and Bayesian Updates) allow strategies to evolve over time.
- **Beautiful Visualizations** show the network health and strategy outcomes at each simulation step.

---

## 🤝 Contributing

Feel free to fork this repo, suggest improvements, or contribute!  
This project is open to new ideas — especially around:
- New attacker/defender strategies
- Advanced RL algorithms for strategy optimization
- More complex wireless models (MIMO, directional antennas)

---

## 📜 License

This project is licensed under the MIT License.

---

## ☕ Support the Developer

If you find this project useful, consider supporting me:

[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=umaryaksambi&button_colour=FFDD00&font_colour=000000&font_family=Comic&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/umaryaksambi)

---

# 🔥 Acknowledgments

- Inspired by research in **Wireless Network Security** and **Game Theory applications**.
- Thanks to open-source Python libraries like **NetworkX**, **Streamlit**, **Matplotlib**, and **Pandas**!
