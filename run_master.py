import subprocess
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Find the python executable in .venv if it exists
if os.name == 'nt': # Windows
    PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
else: # Unix/Mac
    PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

def run_command(command_list):
    """Executes a command and returns the result."""
    print(f"\n>>> Running: {' '.join(command_list)}")
    start = time.time()
    try:
        result = subprocess.run(command_list, check=True)
        end = time.time()
        print(f">>> Completed in {end - start:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f">>> Error executing command: {e}")
        return False

def main():
    datasets = ["jigsaw", "steam"]
    
    print("=" * 60)
    print("  Master Experiment Orchestrator")
    print("  Running Benchmarks, Plotting, and T-SNE for both datasets")
    print("=" * 60)

    # 1. Run Benchmarks
    for ds in datasets:
        print(f"\n--- Benchmarking {ds.upper()} dataset ---")
        success = run_command([PYTHON_EXE, "experiments/run_benchmark.py", "--dataset", ds])
        if not success:
            print(f"Skipping further analysis for {ds} due to benchmark failure.")
            continue

        # 2. Run Plotting
        print(f"\n--- Plotting results for {ds.upper()} ---")
        run_command([PYTHON_EXE, "analysis/plot_results.py", "--dataset", ds])

        # 3. Run T-SNE Visualizer
        print(f"\n--- Generating T-SNE plots for {ds.upper()} ---")
        run_command([PYTHON_EXE, "analysis/tsne_visualizer.py", "--dataset", ds])

    print("\n" + "=" * 60)
    print("  All experiments completed!")
    print("  Results available in:")
    print("    - JSON: results/benchmark_jigsaw.json, results/benchmark_steam.json")
    print("    - Plots: results/plots/jigsaw/, results/plots/steam/")
    print("=" * 60)

if __name__ == "__main__":
    main()
