# test_agent.py
import gymnasium as gym
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from tensorflow.keras.models import load_model

# Inisialisasi environment dengan render biar simulasi muncul
env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen
agent = DQNAgent(state_size, action_size)

# ✅ Load model hasil training kalau ada
try:
    agent.model = load_model("results/cartpole_dqn.h5")
    print("✅ Model results/cartpole_dqn.h5 berhasil dimuat untuk testing")
except:
    print("⚠️ Tidak menemukan model, agen akan pakai bobot random")

# Minimalkan eksplorasi saat testing
agent.epsilon = 0.01  

# Siapkan folder results untuk log
os.makedirs("results", exist_ok=True)
test_scores = []

# Uji selama 5 episode
for e in range(1000):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(500):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = np.reshape(next_state, [1, state_size])
        state = next_state

        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            test_scores.append(time)
            break

env.close()

# ✅ Simpan hasil testing ke HDF5
log_path = os.path.join("results", "test_log.h5")
with h5py.File(log_path, "w") as f:
    f.create_dataset("scores", data=np.array(test_scores))

print(f"✅ Log testing tersimpan di {log_path}")

# ✅ Visualisasi hasil testing
plt.plot(test_scores, marker="o")
plt.title("Hasil Testing DQN pada CartPole")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.grid(True)
plt.show()
