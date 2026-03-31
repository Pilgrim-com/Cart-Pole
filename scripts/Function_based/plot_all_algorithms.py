import os
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, n=100):
    if len(a) < n:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    # Change this if you trained on a different task name, e.g., 'Stabilize-Isaac-Cartpole-v0'
    task_name = "Isaac-Cartpole-v0"
    base_dir = os.path.join("model", task_name)
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Have you trained any models yet?")
        return

    algorithms = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Efficiency Comparison ({task_name})")
    plt.xlabel("Episode")
    plt.ylabel("Duration (Smoothed)")
    
    valid_plots = 0
    for algo in algorithms:
        npy_path = os.path.join(base_dir, algo, f"{algo}_durations.npy")
        if os.path.exists(npy_path):
            durations = np.load(npy_path)
            # Plot smoothed durations
            smoothed = moving_average(durations, n=100)
            plt.plot(np.arange(len(smoothed)) + 100, smoothed, label=algo, linewidth=2)
            valid_plots += 1
        else:
            print(f"Warning: No durations.npy found for {algo} at {npy_path}")
    
    if valid_plots > 0:
        plt.legend()
        plt.grid(True, alpha=0.3)
        output_path = os.path.join(base_dir, "unified_training_curves.png")
        plt.savefig(output_path, dpi=300)
        print(f"\nSUCCESS! Saved unified comparison plot to: {output_path}")
        plt.show()
    else:
        print("No valid data to plot.")

if __name__ == '__main__':
    main()
