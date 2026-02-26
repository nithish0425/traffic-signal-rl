import streamlit as st
import numpy as np
import random

st.set_page_config(page_title="RL Traffic Signal Controller", layout="centered")

st.title("🚦 Reinforcement Learning Based Traffic Signal Controller")
st.write("Simulation of intelligent traffic signal using Q-Learning")

# Traffic density levels
levels = ["Low", "Medium", "High"]

# Actions
actions = ["North-South Green", "East-West Green"]

# Initialize Q-table
Q = np.zeros((3, 3, 2))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def reward(ns, ew, action):
    if action == 0:
        return ew - ns
    else:
        return ns - ew

# Training
for _ in range(500):
    ns = random.randint(0, 2)
    ew = random.randint(0, 2)

    if random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        action = np.argmax(Q[ns][ew])

    r = reward(ns, ew, action)

    next_ns = max(0, ns - 1) if action == 0 else min(2, ns + 1)
    next_ew = max(0, ew - 1) if action == 1 else min(2, ew + 1)

    Q[ns][ew][action] += alpha * (r + gamma * np.max(Q[next_ns][next_ew]) - Q[ns][ew][action])

st.subheader("🚘 Select Traffic Density")

ns_density = st.selectbox("North-South Traffic", levels)
ew_density = st.selectbox("East-West Traffic", levels)

ns = levels.index(ns_density)
ew = levels.index(ew_density)

best_action = np.argmax(Q[ns][ew])

st.subheader("✅ Signal Decision")

if best_action == 0:
    st.success("🟢 GREEN for North-South | 🔴 RED for East-West")
else:
    st.success("🟢 GREEN for East-West | 🔴 RED for North-South")

st.caption("Mini Project | Reinforcement Learning | Traffic Signal Control")
