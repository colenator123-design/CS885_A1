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
**Author:** [Your Name]
**Student ID:** [Your Student ID]