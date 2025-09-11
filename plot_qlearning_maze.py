# -*- coding: utf-8 -*-
import json, os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = "figs/qlearning_maze_results.json"
    if not os.path.exists(path):
        raise FileNotFoundError("找不到結果 JSON，請先執行 TestRLmaze.py")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"]
    curves = data["curves"]
    plt.figure()
    for eps, ys in curves.items():
        plt.plot(range(episodes), ys, label=f"epsilon={eps}")
    plt.xlabel("Episode #")
    plt.ylabel("平均折扣回饋（每回合 100 步，100 次試驗平均）")
    plt.legend()
    os.makedirs("figs", exist_ok=True)
    out = "figs/qlearning_maze_eps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("已輸出圖檔：", out)
