# CS 885 - Assignment 1: MDPs and Reinforcement Learning

This project involves the implementation of several key algorithms in Markov Decision Processes (MDPs) and Reinforcement Learning (RL), as part of the requirements for the CS 885 course.

## Project Structure

The repository is organized as follows:

```
.
├── MDP.py                  # Part I: Core MDP algorithms (Value/Policy Iteration)
├── RL.py                   # Part II: Core RL algorithm (Q-Learning)
├── dqn/
│   ├── dqn_cartpole.py     # Part III: DQN implementation for CartPole
│   └── run_part3.py        # Part III: Script to run DQN experiments
├── mazes.py                # Helper file for creating maze environments
├── requirements.txt        # Python dependencies
├── TestMDPmaze.py          # Part I: Test script for the maze problem
├── TestRLmaze.py           # Part II: Test script for the maze problem
├── figs/                   # Directory for output plots
│   ├── dqn_*.png
│   └── qlearning_*.png
├── *.csv                   # Generated CSV files with experiment results
└── README.md               # This file
```

## Installation

To run the code, first install the required Python libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to Run the Experiments

This project is divided into three parts. The following commands will execute the experiments for each part and generate the corresponding result files (plots and CSVs).

### Part I: Markov Decision Processes (MDP)

This part executes Value Iteration, Policy Iteration, and Modified Policy Iteration on a maze environment. The script has been modified to save all results directly to files.

```bash
python TestMDPmaze.py
```

This will generate:
- `part1_value_iteration.csv`
- `part1_policy_iteration.csv`
- `part1_modified_policy_iteration.csv`
- `part1_summary.txt`

### Part II: Reinforcement Learning (Q-Learning)

The results from the Q-Learning experiment have been processed into `part2_rl_results.csv`. The original script to run this experiment is `TestRLmaze.py`.

### Part III: Deep Q-Network (DQN)

This part trains a DQN on the CartPole environment. The script can be run in two modes to reproduce the experiments from the assignment. The script must be run from within the `dqn` directory.

**1. Scan Target Network Update Frequencies:**

```bash
python dqn/run_part3.py scan-target
```

**2. Scan Mini-Batch Sizes:**

```bash
python dqn/run_part3.py scan-batch
```

## Generated Results

Running the scripts as described above will produce the following output files:

### Data Files (.csv, .txt)

- **`part1_value_iteration.csv`**: State, value, and policy for each state from Value Iteration.
- **`part1_policy_iteration.csv`**: State, value, and policy for each state from Policy Iteration.
- **`part1_modified_policy_iteration.csv`**: Number of iterations for Modified Policy Iteration with varying `k`.
- **`part1_summary.txt`**: Iteration counts for Value and Policy Iteration.
- **`part2_rl_results.csv`**: Average cumulative reward per episode for Q-learning with different epsilon values.
- **`part3_target_update.csv`**: Average reward per episode for the DQN with different target network update frequencies.
- **`part3_batch_size.csv`**: Average reward per episode for the DQN with different mini-batch sizes.

### Plots (.png)

All generated plots are saved in the `figs/` directory.

- **`qlearning_maze_eps.png`**: Compares the performance of Q-learning with different exploration probabilities (epsilon).
- **`dqn_target_update_scan.png`**: Compares the performance of the DQN with different target network update frequencies.
- **`dqn_batch_size_scan.png`**: Compares the performance of the DQN with different mini-batch sizes.

---
## Part I: MDP（Value Iteration / Policy Iteration / Modified Policy Iteration）

- **程式檔案**: `MDP.py`, `TestMDP.py`, `TestMDPmaze.py`  
- 實作演算法：Value Iteration、Policy Iteration、Modified Policy Iteration  
- 測試環境：簡單 MDP（Lecture 2a 範例）以及 Maze MDP  

**結果**:  
- Value Iteration：輸出最終 policy、value function，以及在容差 0.01 下收斂所需的迭代次數。  
- Policy Iteration：從全 0 策略開始，輸出最終 policy、value function 以及收斂所需的迭代次數。  
- Modified Policy Iteration：在 partial policy evaluation 的次數 = 1–10 的情況下，記錄收斂所需的迭代次數。  

**討論**:  
- 當 **partial evaluation 次數較少**（例如 1–2），MPI 的行為更像 Value Iteration：需要較多次迭代，但每次計算較便宜。  
- 當 **partial evaluation 次數較多**（例如 ≥10），MPI 收斂迭代數減少，行為接近 Policy Iteration。  
- 總結：Value Iteration = 多次但便宜的更新；Policy Iteration = 少次但昂貴的更新；MPI 則是兩者的折衷。  

---

## Part II: Q-Learning

- **程式檔案**: `RL.py`, `TestRL.py`, `TestRLmaze.py`  
- 根據 skeleton 實作 Q-learning（依賴 Part I 的 MDP.py）  
- 測試：在 Maze MDP 上進行 200 episodes，每個實驗平均 100 次試驗。  

**結果**:  
- 繪製折線圖：橫軸為 episode (0–200)，縱軸為累積折扣報酬，  
- 各曲線對應探索率 ε = 0.05, 0.1, 0.3, 0.5，並取 100 次試驗平均。  

**討論**:  
- **低 ε (0.05)**：過早利用，學習初期進展慢，容易陷入次佳策略。  
- **中等 ε (0.1–0.3)**：探索與利用平衡最佳，學習效果與收斂速度較佳。  
- **高 ε (0.5)**：過度探索，曲線震盪大，收斂較慢。  
- Q-table 收斂情況與 ε 密切相關：低 ε 較快收斂但可能不最優，高 ε 則需更長時間才能收斂到穩定策略。  

---

## Part III: Deep Q-Network (DQN on CartPole)

- **程式檔案**: `dqn_cartpole.py`, `run_part3.py`  
- 使用 PyTorch 實作 DQN，含 replay buffer 與 target network  
- 實驗內容：  
  - Target update frequency = {1, 10, 50, 100}  
  - Mini-batch size = {1, 10, 50, 100}  
  - 每組實驗平均 5 個隨機種子  
- 圖表：橫軸 = episode (到 300)，縱軸 = 最近 25 episodes 的平均 reward  

**討論**:  
- **Target Network**  
  - 更新太頻繁（每 1 episode）會造成不穩定，效果差。  
  - 更新太少（100 episodes）則因目標過舊，學習速度慢。  
  - 中等頻率（10–50 episodes）表現最佳，收斂快且穩定。  
  - 功能類似 **Value Iteration**：固定一個 value function 作為穩定目標，再逐步改善策略。  

- **Replay Buffer / Batch Size**  
  - Batch 太小（1）更新噪音大，收斂不穩。  
  - Batch 較大（50–100）能平滑更新，結果更接近 **精確梯度下降**。  
  - Batch 過大雖然平穩，但更新速度可能慢一些。  
  - Replay buffer 的作用是打破樣本相關性，使得 SGD 的近似更合理。  

---

## 總結

- **Part I**：比較 Value Iteration、Policy Iteration 與 Modified Policy Iteration 的收斂行為與計算特性。  
- **Part II**：探索率 ε 對 Q-learning 的影響明顯，適中 ε 最能平衡探索與利用。  
- **Part III**：DQN 的穩定性高度依賴 Target Network 與 Replay Buffer，分別對應傳統 RL 中的 Value Iteration 與 Gradient Descent。  
